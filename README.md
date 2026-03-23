# Pluto 训练 — EKS 8x H200 (p5en.48xlarge)

[Pluto](https://github.com/jchengai/pluto) 自动驾驶规划模型，适配 H200 GPU + EKS HyperPod。

## 文件说明

```
├── Dockerfile                       # NATTEN 源码编译 (sm_90) + nuplan-devkit + pluto
├── entrypoint.sh                    # cache / train 阶段，含 stdout 日志补丁
├── build-and-push.sh                # 构建镜像并推送到 ECR
├── 1-k8s-nuplan-download.yaml       # 下载 nuPlan 数据集
├── 2-k8s-nuplan-reorganize.yaml     # 整理 NVMe 上的数据
├── 3-pluto-cache-job-kr.yaml        # 特征缓存（仅 CPU）
└── 4-pluto-train-job.yaml           # 训练任务（8x H200）
```

## 快速开始

```bash
# 1. 构建镜像
bash build-and-push.sh

# 2. 数据准备（一次性，按顺序执行）
kubectl apply -f 1-k8s-nuplan-download.yaml
kubectl apply -f 2-k8s-nuplan-reorganize.yaml
kubectl apply -f 3-pluto-cache-job-kr.yaml

# 3. 启动训练
kubectl apply -f 4-pluto-train-job.yaml
kubectl logs -f job/pluto-train
```

## 关键配置（4-pluto-train-job.yaml）

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| BATCH_SIZE | 64 | 每卡 batch size |
| EPOCHS | 25 | |
| PRECISION | 32 | 稳定后可改 `bf16-mixed`，约 2 倍加速 |
| NPROC_PER_NODE | 8 | GPU 数量 |
| NUM_WORKERS | 8 | 每个 rank 的 DataLoader workers（8 rank x 8 = 64） |

## 踩坑记录

- **NATTEN**：必须源码编译 `NATTEN_CUDA_ARCH="9.0"`，PyPI 的 wheel 只有 CPU 版本
- **DDP**：必须用 `ddp_find_unused_parameters_true`，`false` 会导致 8 卡 NCCL 死锁
- **RichProgressBar**：kubectl logs 非 TTY 环境下不显示，entrypoint 中已替换为 stdout callback
- **np.bool / compute_on_step**：Dockerfile 中已修补，兼容 numpy 1.24+ / torchmetrics 1.0+
