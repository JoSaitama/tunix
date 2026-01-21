# Tunix / my_example 环境配置流程（Ubuntu 22.04 + Python 3.11 + TPU）

> 目标：从拉取代码开始，完成本地虚拟环境与依赖安装，并能运行 `./my_example/run_grpo_gemma.sh`。

## 1) 拉代码

```bash
git clone -b my-changes --single-branch https://github.com/yangziao56/tunix.git
cd tunix
```

## 2) 创建并激活虚拟环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## 3) 安装依赖（开发模式）

```bash
pip install -U pip
pip install -e ".[dev]"
```

## 4) 配置 Hugging Face / Kaggle 凭据（my_example/.env）

`my_example/.env` 已被 `.gitignore` 忽略（不会被提交），按需填写：

```bash
cat > my_example/.env <<'EOF'
HF_TOKEN=YOUR_HF_TOKEN
KAGGLE_USERNAME=YOUR_KAGGLE_USERNAME
KAGGLE_KEY=YOUR_KAGGLE_KEY
EOF

chmod 600 my_example/.env
```

## 5) TPU 环境：安装 TPU 版 JAX（解决 device_count=1 / mesh_shape 报错）

如果运行时出现：

- `A Google TPU may be present... Falling back to cpu.`
- `ValueError: Number of devices 1 must be >= the product of mesh_shape (4, 1)`

说明当前是 CPU 后端（`jax.device_count()==1`）。在 TPU VM 上执行：

```bash
pip uninstall -y jax jaxlib
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

验证：

```bash
python -c "import jax; print('backend:', jax.default_backend()); print('device_count:', jax.device_count()); print('devices:', jax.devices())"
```

期望看到 `backend: tpu` 且 `device_count: 4`（或与你的 TPU 拓扑一致）。

（可选）你也可以用系统设备节点快速确认 TPU 是否可见：

```bash
ls -l /dev/accel*
```

在 TPU VM 上通常会看到类似 `/dev/accel0 ... /dev/accel3`（数量取决于 TPU 拓扑）；如果提示 `No such file or directory`，通常说明当前不是 TPU VM，或 TPU 运行时/驱动未正确安装。

## 6) 安装 gcsfs（解决 gs:// tokenizer 读取失败）

如果报错 `ImportError: Please install gcsfs to access Google Storage`：

```bash
pip install -U gcsfs
```

注意：`pip` 可能提示 `datasets` 与 `fsspec` 版本冲突；如需消除冲突，建议将 `gcsfs/fsspec` 降到与当前 `datasets` 兼容的版本（按实际输出选择版本号），例如：

```bash
pip install "gcsfs==2025.10.0" "fsspec==2025.10.0"
```

## 7) 运行验证

最小跑通（示例）：

```bash
./my_example/run_grpo_gemma.sh
```

带日志与 checkpoint（项目验收用法之一）：

```bash
./my_example/run_grpo_gemma.sh \
  --num-test-batches 1 \
  --metrics-log-dir /tmp/content/tmp/tensorboard/grpo_$(date +%Y%m%d_%H%M%S) \
  --checkpoint-root /tmp/content/ckpts_run2_$(date +%Y%m%d_%H%M%S)
```

> 若遇到 Hugging Face 401（`GatedRepoError`），需要在 HF 网页端完成模型权限/条款确认，并确保 `HF_TOKEN` 具备访问权限。
