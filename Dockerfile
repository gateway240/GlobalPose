FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    cmake \
    ninja-build \
    pkg-config \
    pybind11-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    freeglut3-dev \
    libboost-all-dev \
    libeigen3-dev \
    libfmt-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgl1 \
    libglfw3 \
    libglfw3-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

# Workspace
WORKDIR /app
# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Put the virtual environment OUTSIDE the mounted source tree
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# --------------------------------------------------------------------------
# Build g2o
# --------------------------------------------------------------------------
# https://github.com/RainerKuemmerle/g2o/releases/tag/20241228_git
ARG G2O_COMMIT=eec325a1da1273e87bc97887d49e70570f28570c
RUN git clone https://github.com/RainerKuemmerle/g2o.git /tmp/g2o && \
    cd /tmp/g2o && \
    git checkout ${G2O_COMMIT}
    
RUN cmake -S /tmp/g2o -B /tmp/g2o/build \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DG2O_BUILD_APPS=OFF \
        -DG2O_BUILD_EXAMPLES=OFF \
        -DG2O_BUILD_UNITTESTS=OFF && \
    cmake --build /tmp/g2o/build -j$(nproc) && \
    cmake --install /tmp/g2o/build && \
    ldconfig && \
    rm -rf /tmp/g2o

# --------------------------------------------------------------------------
# Build Sophus
# --------------------------------------------------------------------------
ARG SOPHUS_COMMIT=d0b7315a0d90fc6143defa54596a3a95d9fa10ec
RUN git clone https://github.com/strasdat/Sophus.git /tmp/Sophus && \
    cd /tmp/Sophus && \
    git checkout ${SOPHUS_COMMIT}

RUN cmake -S /tmp/Sophus -B /tmp/Sophus/build \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release  \
        -DBUILD_SOPHUS_TESTS=OFF && \
    cmake --build /tmp/Sophus/build -j$(nproc) && \
    cmake --install /tmp/Sophus/build && \
    rm -rf /tmp/Sophus

# --------------------------------------------------------------------------
# Build carticulate
# --------------------------------------------------------------------------
ARG CARTICULATE_COMMIT=558526b86f2440d8785914d837b0d5eccd1a16da

RUN git clone https://github.com/gateway240/carticulate.git /tmp/carticulate && \
    cd /tmp/carticulate && \
    git checkout ${CARTICULATE_COMMIT}

RUN cmake -S /tmp/carticulate \
          -B /tmp/carticulate/build \
          -GNinja \
          -DPython_EXECUTABLE=/opt/venv/bin/python && \
    cmake --build /tmp/carticulate/build -j$(nproc)

# Install into the active UV environment
RUN SITEPKG=$(/opt/venv/bin/python -c "import site; print(site.getsitepackages()[0])") && \
    mkdir -p "$SITEPKG/carticulate" && \
    cp /tmp/carticulate/scripts/__init__.py "$SITEPKG/carticulate/" && \
    cp /tmp/carticulate/scripts/utils.py "$SITEPKG/carticulate/" && \
    cp /tmp/carticulate/build/carticulate*.so "$SITEPKG/carticulate/" && \
    ls -la "$SITEPKG/carticulate"

# pybind11-stubgen
RUN uv pip install \
    --python /opt/venv/bin/python \
    pybind11-stubgen
# Generate carticulate.pyi
RUN SITEPKG=$(/opt/venv/bin/python -c "import site; print(site.getsitepackages()[0])") && \
    cd /tmp && \
    PYTHONPATH="$SITEPKG" /opt/venv/bin/pybind11-stubgen carticulate -o /tmp/stubs && \
    cp /tmp/stubs/carticulate/__init__.pyi "$SITEPKG/carticulate/carticulate.pyi" && \
    ls -la "$SITEPKG/carticulate" 

# Copy source
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

CMD ["/bin/bash"]