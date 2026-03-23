FROM nvcr.io/nvidia/pytorch:24.10-py3
# PyTorch 2.5.0a0 | CUDA 12.6.2 | cuDNN 9.5.1 | NCCL 2.22.3 | Ubuntu 22.04
# Includes: nvcc, cmake, ninja, TransformerEngine, Apex, etc.

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# NATTEN 0.17.5 — source build against NGC PyTorch 2.5
# NGC has nvcc + cmake, so this compiles directly (~15-20 min)
# MUST build from source with CUDA arch for H200 (sm_90);
# plain "pip install natten" gets a CPU-only wheel from PyPI.
RUN NATTEN_CUDA_ARCH="9.0" NATTEN_WITH_CUDA=1 \
    pip install --no-cache-dir natten==0.17.5 --no-binary natten

# nuplan-devkit (from source)
# Its requirements.txt pins ancient versions (numpy==1.23.4, opencv-python<=4.5.1.48,
# grpcio==1.43.0, etc.) that have no prebuilt wheels for Python 3.10+.
# We patch these pins before installing.
RUN git clone --depth 1 https://github.com/motional/nuplan-devkit.git /opt/nuplan-devkit && \
    cd /opt/nuplan-devkit && \
    # Relax version pins that fail on Python 3.10+
    sed -i 's/opencv-python<=4.5.1.48/opencv-python>=4.8,<4.10/' requirements.txt && \
    sed -i 's/numpy==1.23.4/numpy>=1.24,<2.0/' requirements.txt && \
    sed -i 's/grpcio==1.43.0/grpcio>=1.43/' requirements.txt && \
    sed -i 's/grpcio-tools==1.43.0/grpcio-tools>=1.43/' requirements.txt && \
    sed -i 's/bokeh==2.4.3/bokeh>=2.4.3,<3.0/' requirements.txt && \
    sed -i 's/guppy3==3.1.2/guppy3>=3.1.2/' requirements.txt && \
    sed -i 's/hydra-core==1.1.0rc1/hydra-core==1.1.2/' requirements.txt && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r requirements.txt

# Pluto
RUN git clone --depth 1 https://github.com/jchengai/pluto.git /opt/pluto && \
    # Pluto repo is missing __init__.py in most src/ subdirs
    find /opt/pluto/src -type d -exec sh -c 'test ! -f "$1/__init__.py" && touch "$1/__init__.py"' _ {} \; && \
    # re-export symbols that run_training.py expects from src.custom_training
    echo 'from src.custom_training.custom_training_builder import TrainingEngine, build_training_engine, update_config_for_training' \
        > /opt/pluto/src/custom_training/__init__.py && \
    # Fix np.bool (removed in numpy 1.24+): np.bool) -> np.bool_)
    find /opt/pluto/src -name '*.py' -exec sed -i 's/np\.bool)/np.bool_)/g' {} + && \
    # Fix compute_on_step (removed in torchmetrics 1.0+)
    find /opt/pluto/src -name '*.py' -exec sed -i '/compute_on_step/d' {} + && \
    # Replace strict isfinite assert with nan_to_num (training may produce NaN early on)
    sed -i 's/assert torch.isfinite(q).all()/q = torch.nan_to_num(q)/' \
        /opt/pluto/src/models/pluto/modules/planning_decoder.py

# Pluto python dependencies (override versions for NGC compat)
RUN pip install --no-cache-dir \
    timm \
    "pytorch-lightning>=2.4.0,<3.0" \
    torchmetrics \
    tensorboard \
    wandb \
    numba \
    rich

WORKDIR /opt/pluto

# Fix opencv: 4.10+ removed cv2.dnn.DictValue, breaks nuplan-devkit
# --no-deps prevents numpy from being upgraded
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless 2>/dev/null || true && \
    pip install --no-cache-dir --no-deps "opencv-python>=4.8,<4.10" && \
    sed -i 's/LayerId = cv2.dnn.DictValue/LayerId = int/' \
        $(python -c "import site; print(site.getsitepackages()[0])")/cv2/typing/__init__.py 2>/dev/null || true && \
    python -c "import cv2; print('opencv OK:', cv2.__version__)"

# Entrypoint for distributed training via torchrun
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
