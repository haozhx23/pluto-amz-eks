#!/bin/bash
set -euo pipefail

echo "=== Pluto Training Entrypoint ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi || true

# --- NCCL tuning for H200 + EFA ---
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_PROTO=${NCCL_PROTO:-simple}
export FI_EFA_USE_DEVICE_RDMA=${FI_EFA_USE_DEVICE_RDMA:-1}
export FI_PROVIDER=${FI_PROVIDER:-efa}
export FI_EFA_FORK_SAFE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}

# --- Resolve distributed training env from K8s ---
# For single-node, torchrun handles everything.
# For multi-node, MASTER_ADDR / MASTER_PORT must be set by the Job spec.
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Distributed config: nnodes=$NNODES, node_rank=$NODE_RANK, nproc=$NPROC_PER_NODE, master=$MASTER_ADDR:$MASTER_PORT"

# --- nuPlan data paths ---
export NUPLAN_DATA_ROOT=${NUPLAN_DATA_ROOT:-/nvme/nuplan}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/nvme/nuplan/maps}
echo "Data root: $NUPLAN_DATA_ROOT"
echo "Maps root: $NUPLAN_MAPS_ROOT"

# Verify data is accessible
if [ ! -d "$NUPLAN_DATA_ROOT/data/cache" ]; then
    echo "WARNING: $NUPLAN_DATA_ROOT/data/cache not found. Check volume mounts."
fi
if [ ! -d "$NUPLAN_MAPS_ROOT" ]; then
    echo "WARNING: $NUPLAN_MAPS_ROOT not found. Check volume mounts."
fi

cd /opt/pluto
export PYTHONPATH="/opt/pluto:${PYTHONPATH:-}"
export NATTEN_LOG_LEVEL=critical

# --- Fix: Replace RichProgressBar with stdout-friendly logging ---
# RichProgressBar uses terminal control chars that are invisible in kubectl logs.
cat > /tmp/stdout_progress.py << 'PYEOF'
import pytorch_lightning as pl
import time
import torch

class StdoutProgressCallback(pl.Callback):
    def __init__(self):
        self._epoch_t = None

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        self._epoch_t = time.time()
        total = trainer.num_training_batches
        print(f"[Epoch {trainer.current_epoch+1}/{trainer.max_epochs}] Starting — {total} steps", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank != 0:
            return
        if batch_idx % 50 == 0 or batch_idx == 1:
            # Extract loss from training_step outputs
            loss = "N/A"
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = f"{outputs['loss'].item():.4f}"
            elif isinstance(outputs, torch.Tensor):
                loss = f"{outputs.item():.4f}"
            total = trainer.num_training_batches
            pct = 100.0 * batch_idx / max(total, 1)
            print(f"  step {batch_idx}/{total} ({pct:.0f}%) loss={loss}", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        elapsed = time.time() - self._epoch_t if self._epoch_t else 0
        metrics = {k: f"{v:.4f}" if hasattr(v, "__float__") else str(v)
                   for k, v in trainer.callback_metrics.items()}
        print(f"[Epoch {trainer.current_epoch+1}] Done in {elapsed:.0f}s | {metrics}", flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        metrics = {k: f"{v:.4f}" if hasattr(v, "__float__") else str(v)
                   for k, v in trainer.callback_metrics.items()}
        print(f"[Validation] {metrics}", flush=True)
PYEOF

# Patch training builder: swap RichProgressBar → StdoutProgressCallback
BUILDER=/opt/pluto/src/custom_training/custom_training_builder.py
sed -i 's/RichProgressBar()/StdoutProgressCallback()/' "$BUILDER"
sed -i '/from.*RichProgressBar/d' "$BUILDER"
sed -i '1i import sys; sys.path.insert(0, "/tmp"); from stdout_progress import StdoutProgressCallback' "$BUILDER"
echo "Patched RichProgressBar → StdoutProgressCallback"

# --- Phase selection ---
PHASE=${PHASE:-train}

if [ "$PHASE" = "cache" ]; then
    echo "=== Running feature caching ==="
    CACHE_PATH=${CACHE_PATH:-/data/cache_pluto}

    python run_training.py \
        py_func=cache +training=train_pluto \
        scenario_builder=nuplan \
        cache.cache_path=${CACHE_PATH} \
        cache.cleanup_cache=${CACHE_CLEANUP:-false} \
        scenario_filter=${SCENARIO_FILTER:-training_scenarios_1M} \
        worker.threads_per_node=${CACHE_WORKERS:-40} \
        ${EXTRA_ARGS:-}

    # Validate cache is not empty before proceeding
    CACHE_COUNT=$(find "${CACHE_PATH}" -name "*.gz" -o -name "*.pkl" 2>/dev/null | wc -l)
    echo "Cache files generated: ${CACHE_COUNT}"
    if [ "$CACHE_COUNT" -eq 0 ]; then
        echo "ERROR: Cache is empty! Check data paths and scenario_filter."
        exit 1
    fi

elif [ "$PHASE" = "train" ]; then
    echo "=== Running training ==="
    # GPU monitor in background
    nvidia-smi dmon -s umt -d 10 > /tmp/gpu_monitor.log 2>&1 &

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} python run_training.py \
        py_func=train +training=train_pluto \
        worker=single_machine_thread_pool worker.max_workers=32 \
        scenario_builder=nuplan \
        cache.cache_path=${CACHE_PATH:-/data/cache_pluto} \
        cache.use_cache_without_dataset=true \
        data_loader.params.batch_size=${BATCH_SIZE:-64} \
        data_loader.params.num_workers=${NUM_WORKERS:-16} \
        lr=${LR:-1e-3} \
        epochs=${EPOCHS:-25} \
        warmup_epochs=${WARMUP_EPOCHS:-3} \
        weight_decay=${WEIGHT_DECAY:-0.0001} \
        lightning.trainer.params.accelerator=gpu \
        lightning.trainer.params.devices=${NPROC_PER_NODE:-8} \
        lightning.trainer.params.strategy=ddp_find_unused_parameters_true \
        lightning.trainer.params.precision=${PRECISION:-bf16-mixed} \
        wandb.mode=${WANDB_MODE:-disabled} \
        wandb.project=${WANDB_PROJECT:-pluto} \
        wandb.name=${WANDB_NAME:-pluto-h200} \
        ${EXTRA_ARGS:-}

    kill $(jobs -p) 2>/dev/null || true
else
    echo "Unknown PHASE=$PHASE. Use 'cache' or 'train'."
    exit 1
fi
