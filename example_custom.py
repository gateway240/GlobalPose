r"""
Test the system with an example IMU measurement sequence.
"""
from pathlib import Path

import itertools
from articulate.math import quaternion_to_rotation_matrix
from scipy.spatial.transform import Rotation
from articulate import SMPLJoint
import quaternion


import torch
from net import GPNet
import numpy as np
import os
import articulate as art


DATA_DIR = Path("./data/examples")
# motion = "squats_deep"
# frame_start = 0
# frame_end = 400
# frame_end = -1

motion = "walking"
frame_start = 0
frame_end = 1000

# motion = "jogging"
# frame_start = 100
# frame_end = 1200

# motion = "half_jacks"
# frame_start = 0
# frame_end = 400

# motion = "curls"
# frame_start = 0
# frame_end = 1000

# motion = "box_jabs"
# frame_start = 0
# frame_end = 1000
# Person starting facing left (squats) or maybe all??
# R_fix = torch.tensor([
#     [ 0, -1, 0],
#     [ 1,  0, 0],
#     [ 0,  0, 1],
# ], dtype=torch.float32)
# Person starting facing right(treadmill)
# R_fix = torch.tensor([
#     [0, 1, 0],
#     [1, 0, 0],
#     [0, 0,-1],
# ], dtype=torch.float32)

R_arm = torch.tensor([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0,1],
], dtype=torch.float32)
R_body = torch.tensor([
    [ 0, -1, 0],
    [ 1,  0, 0],
    [ 0,  0, 1],
], dtype=torch.float32)
R_identity = torch.tensor([
    [ 1, 0, 0],
    [ 0,  1, 0],
    [ 0,  0, 1],
], dtype=torch.float32)


PELVIS_IDX = 5
CAL_FRAMES = 60  # 1 sec at 60 FPS

tracked_joints = [
    SMPLJoint.LELBOW,
    SMPLJoint.RELBOW,
    SMPLJoint.LKNEE,
    SMPLJoint.RKNEE,
    SMPLJoint.HEAD,
    SMPLJoint.PELVIS,
]


def generate_coordinate_frame_candidates():
    """
    Generate every proper orthogonal axis mapping.

    Returns:
        [(name, R), ...]
    """

    candidates = []

    axes = [
        torch.tensor([1, 0, 0], dtype=torch.float32),
        torch.tensor([0, 1, 0], dtype=torch.float32),
        torch.tensor([0, 0, 1], dtype=torch.float32),
    ]

    for perm in itertools.permutations(range(3)):
        P = torch.stack(
            [axes[perm[0]], axes[perm[1]], axes[perm[2]]],
            dim=1,
        )

        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    S = torch.diag(
                        torch.tensor(
                            [sx, sy, sz],
                            dtype=torch.float32,
                        )
                    )

                    R = P @ S

                    det = torch.det(R)

                    #
                    # Only keep proper rotations
                    #
                    if torch.isclose(
                        det,
                        torch.tensor(1.0),
                        atol=1e-5,
                    ):
                        name = f"P{perm}_sx{sx:+d}_sy{sy:+d}_sz{sz:+d}"

                        candidates.append((name, R))

    print(f"Generated {len(candidates)} candidates")

    return candidates


def compute_heading_correction(
    ori,  # [frames, sensors, 3, 3]
    base_sensor=5,  # Pelvis
    base_heading_axis=(1.0, 0.0, 0.0),
    desired_heading=(1.0, 0.0, 0.0),
    up_axis=(0.0, 0.0, 1.0),
):
    """
    Compute heading correction around Xsens global +Z.

    Assumptions:
      - ori maps sensor coordinates to Xsens global coordinates.
      - Xsens frame is North-West-Up.
      - Pelvis +X points forward.
      - Desired heading is magnetic North (+X).
    """

    device = ori.device
    dtype = ori.dtype

    up = torch.tensor(up_axis, device=device, dtype=dtype)
    up = up / torch.linalg.norm(up)

    sensor_forward = torch.tensor(base_heading_axis, device=device, dtype=dtype)

    target_forward = torch.tensor(desired_heading, device=device, dtype=dtype)

    # First pelvis orientation.
    R_base = ori[0, base_sensor]

    # Pelvis forward direction expressed in global coordinates.
    heading = R_base @ sensor_forward

    # Project both directions onto the horizontal plane.
    heading = heading - torch.dot(heading, up) * up
    target_forward = target_forward - torch.dot(target_forward, up) * up

    heading = heading / torch.linalg.norm(heading)
    target_forward = target_forward / torch.linalg.norm(target_forward)

    # Signed angle from measured heading to desired heading.
    cross = torch.linalg.cross(heading, target_forward)

    sin_angle = torch.dot(up, cross)
    cos_angle = torch.dot(heading, target_forward)

    angle = torch.atan2(sin_angle, cos_angle)

    # Rotation around global vertical axis.
    K = torch.tensor(
        [
            [0.0, -up[2], up[1]],
            [up[2], 0.0, -up[0]],
            [-up[1], up[0], 0.0],
        ],
        device=device,
        dtype=dtype,
    )

    I = torch.eye(3, device=device, dtype=dtype)

    C = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

    return angle, C


def make_calibration_pose(num_frames: int = 1) -> torch.Tensor:
    pose = torch.eye(3).repeat(num_frames, 24, 1, 1)

    left_arm_down = torch.tensor(
        Rotation.from_euler("z", -85, degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    right_arm_down = torch.tensor(
        Rotation.from_euler("z", 85, degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    left_elbow_bend = torch.tensor(
        Rotation.from_euler("xy", [-20,-10], degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    right_elbow_bend = torch.tensor(
        Rotation.from_euler("xy", [-20,10], degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    pose[:, SMPLJoint.LSHOULDER] = left_arm_down
    pose[:, SMPLJoint.RSHOULDER] = right_arm_down

    pose[:, SMPLJoint.LELBOW] = left_elbow_bend
    pose[:, SMPLJoint.RELBOW] = right_elbow_bend

    # left_wrist_bend = torch.tensor(
    #     Rotation.from_euler("x", 30, degrees=True).as_matrix(),
    #     dtype=torch.float32,
    # )

    # right_wrist_bend = torch.tensor(
    #     Rotation.from_euler("x", -30, degrees=True).as_matrix(),
    #     dtype=torch.float32,
    # )

    # pose[:, SMPLJoint.LWRIST] = left_wrist_bend
    # pose[:, SMPLJoint.RWRIST] = right_wrist_bend

    return pose


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = GPNet().to(device)
acc_raw = torch.load( DATA_DIR / f"{motion}_accelerations.pt")
print(acc_raw[0])
print(acc_raw.shape)
ori_raw = torch.load(DATA_DIR / f"{motion}_orientations.pt")

ang_vel_raw = torch.load(DATA_DIR / f"{motion}_angular_velocities.pt")

acc_raw = acc_raw[frame_start:frame_end]
ori_raw = ori_raw[frame_start:frame_end]
ang_vel_raw = ang_vel_raw[frame_start:frame_end]

num_frames = ori_raw.shape[0]


print(ori_raw[0])
print(ori_raw.shape)
pose_cal = make_calibration_pose(1)[0].to(device)  # (24,3,3)

R_bone_cal = pose_cal[tracked_joints].to(device) # (6,3,3)
candidates = generate_coordinate_frame_candidates()
# for name, R_var in candidates
#     print()
#     print("=" * 80)
#     print(name)
#     print("=" * 80)


# Apply candidate transform to experimental IMU data
#
# ori_test = ori @ R_fix
# acc_test = acc @ R_fix
# R_fix = torch.tensor(
#     [
#         [-1, 0, 0],
#         [0, 0, 1],
#         [0, 1, 0],
#     ],
#     dtype=torch.float32,
# )
# R_fix = torch.tensor(
#     [
#         [1, 0, 0],
#         [0, 0, -1],
#         [0, 1, 0],
#     ],
#     dtype=torch.float32,
# )
# All data corrected to new basis

# R_fix = torch.tensor(
#     [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
#     dtype=torch.float32,
# )
# R_fix = torch.tensor(
#     [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
#     dtype=torch.float32,
# )
# R_fix = torch.tensor(
#     [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
#     dtype=torch.float32,
# )
# Good for walking
# R_fix = torch.tensor(
#     [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
#     dtype=torch.float32,
# )
# R_fix = torch.tensor([
#     [ 0,  0, -1],
#     [ 0, -1,  0],
#     [-1,  0,  0],
# ], dtype=torch.float32)
# R_fix = torch.tensor(
#     [[-1,  0,  0],
#  [ 0,  0, -1],
#  [ 0, -1,  0]],
#     dtype=torch.float32,
# )
# R_fix = torch.tensor([
#     [0, 1, 0],
#     [1, 0, 0],
#     [0, 0,-1],
# ], dtype=torch.float32)
ori = torch.stack(
    [
        quaternion_to_rotation_matrix(ori_raw[i])  # (6, 4) -> (6, 3, 3)
        for i in range(num_frames)
    ],
    dim=0,
)
# angle, C_heading = compute_heading_correction(
#     ori,
#     base_sensor=PELVIS_IDX,
# )

# print("Heading correction:", torch.rad2deg(angle).item())
# print("Correction matrix:", C_heading)
R_var = R_body
R_fix = torch.stack([
    R_var,   # radius_l_imu
    R_var,   # radius_r_imu
    R_var,  # tibia_l_imu
    R_var,  # tibia_r_imu
    R_var,  # head_imu
    R_var,  # pelvis_imu
]).to(ori.device)

ori = (ori @ R_fix).to(device)
acc_raw = torch.matmul(
    acc_raw.unsqueeze(-2),
    R_fix.unsqueeze(0)
).squeeze(-2).to(device)

ang_vel_raw = torch.matmul(
    ang_vel_raw.unsqueeze(-2),
    R_fix.unsqueeze(0)
).squeeze(-2).to(device)

# Calibration frame
ori_cal = ori_raw[:CAL_FRAMES].mean(dim=0)
acc_cal = acc_raw[:CAL_FRAMES].mean(dim=0).to(device)
R_ori_cal = quaternion_to_rotation_matrix(ori_cal.reshape(-1, 4)).view(6, 3, 3)
R_ori_cal = (R_ori_cal @ R_fix).to(device)
smpl2imu = R_ori_cal[PELVIS_IDX].T.to(device)
print(smpl2imu)

device2bone = smpl2imu.matmul(R_ori_cal).transpose(1, 2).matmul(R_bone_cal).to(device)
acc_offsets = smpl2imu.matmul(
    acc_cal.unsqueeze(-1)
)  # [num_imus, 3, 1], already in global inertial frame

print(device2bone.shape)
print(acc_offsets.shape)

acc_in = (smpl2imu.matmul(acc_raw.unsqueeze(-1)) - acc_offsets.unsqueeze(0)).squeeze(-1).to(device)
# torch.save(acc_in.cpu(),os.path.join(paths.example_dir, f"{motion}_cal_accelerations.pt"))
ori_in = smpl2imu.matmul(ori).matmul(device2bone).to(device)
# torch.save(ori_in.cpu(),os.path.join(paths.example_dir, f"{motion}_cal_orientations.pt"))
# acc_in = torch.zeros_like(acc_raw)
# x = normalize_and_concat(acc_in, ori_in).to(device)
# pose, tran = net.forward_offline(x)  # offline
# print(acc_in.device)
# print(ori_in.device)
# print(pose_cal.device)
net.rnn_initialize(pose_cal)
pose_prediction = []
tran_prediction = []

for i in range(num_frames):
    pose_i, tran_i = net.forward_frame(
        acc_in[i],
        ang_vel_raw[i],
        ori_in[i]
    )
    pose_prediction.append(pose_i.to(device))
    tran_prediction.append(tran_i.to(device))

pose_prediction = torch.stack(pose_prediction, dim=0).to(device)
tran_prediction = torch.stack(tran_prediction, dim=0).to(device)

prediction = pose_prediction.clone().detach().cpu().numpy()

trans = tran_prediction.clone().detach().cpu().numpy()
# trans = np.repeat(trans_0, n_frames, axis=0)
print("Trans shape:", trans.shape)
print("Prediction shape:", prediction.shape)
# gender ()
gender = "male"
# mocap_framerate ()
mocap_framerate = 60.
# betas (16,)
# From SMPL2AddBiomechanics repo
betas = [
    -0.2627612,
    1.47132885,
    -0.31743591,
    0.32307386,
    0.69473532,
    0.06988167,
    1.15463818,
    -2.03527317,
    1.25517997,
    -0.96329528,
    -0.62346964,
    0.73335359,
    0.54041496,
    0.63366898,
    -0.25407558,
    0.57630826,
]
# dmpls (2751, 8)
# poses (2751, 156)
prediction = np.reshape(prediction, [prediction.shape[0], -1, 3, 3])
prediction = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(prediction))
prediction = np.reshape(prediction, [prediction.shape[0], -1])
poses = prediction
np.savez(
    DATA_DIR / "{}_smpl_pose_globalpose.npz".format(motion),  # noqa: UP032
    trans=trans,
    gender=gender,
    mocap_framerate=mocap_framerate,
    betas=betas,
    poses=poses,
)
R_vis = torch.tensor(
    Rotation.from_euler(
        "xy",
        [180, 0],
        degrees=True,
    ).as_matrix(),
    dtype=torch.float32,
).to(device)
pose_vis = pose_prediction.clone()
tran_vis = tran_prediction.clone()
pose_vis[:, 0] = R_vis @ pose_vis[:, 0]
tran_vis = (R_vis @ tran_vis.unsqueeze(-1)).squeeze(-1)
# print(pose.shape, tran.shape)
# print(pose[0])
# pose, tran = [torch.stack(_) for _ in zip(*[net.forward_online(f) for f in x])]   # online
# art.ParametricModel(paths.smpl_file).view_motion([pose_vis], [tran_vis])
pose_vis = torch.as_tensor(pose_vis, dtype=torch.float32).cpu()
tran_vis = torch.as_tensor(tran_vis, dtype=torch.float32).cpu()

art.ParametricModel('models/SMPL_male.pkl').view_motion(
    [pose_vis],
    [tran_vis]
)