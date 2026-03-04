# Tunix / my_example 环境配置流程（Ubuntu 22.04 + Python 3.11 + TPU）

> 如果你能看到这个文件，说明你已经 `git clone` 了仓库（文件路径：`ENV_SETUP.md`）。
>
> 目标：完成本地虚拟环境与依赖安装，并能运行 `./my_example/run_grpo_gemma.sh`。

## 0) 先确认机器真的有 TPU（最重要）

在 TPU VM 上通常可以看到 `/dev/accel*` 设备节点：

```bash
ls -l /dev/accel*
```

一般会看到类似 `/dev/accel0 ... /dev/accel3`（数量取决于 TPU 拓扑）；如果提示 `No such file or directory`，通常说明当前不是 TPU VM，或 TPU 运行时/驱动未正确安装。

## 1) 拉代码

```bash
git clone -b my-changes --single-branch https://github.com/yangziao56/tunix.git
cd tunix
```

## 2) 创建并激活虚拟环境（`.venv` vs `.venv_jax081`）

当前仓库里常见两套环境名：

- `.venv`：通用开发环境，按 `pip install -e ".[dev]"` 正常安装即可（依赖可能更“新”）。
- `.venv_jax081`：JAX 0.8.1 对齐环境（更贴近 `pyproject.toml` 里的 `jax[tpu] <= 0.8.1` 约束）。

推荐：

- 如果你主要跑当前 `my_example`/`my_example_qwen_aime`，并希望和项目 `prod` 约束保持一致，优先用 `.venv_jax081`。
- 如果你只是日常开发且不要求固定 JAX 版本，用 `.venv` 即可。

二选一执行（不要同时激活）：

```bash
# A) 通用开发环境
python3.11 -m venv .venv
source .venv/bin/activate

# B) JAX 0.8.1 对齐环境（推荐给 TPU 训练复现）
python3.11 -m venv .venv_jax081
source .venv_jax081/bin/activate
```

## 3) 安装依赖（开发模式）

```bash
pip install -U pip
pip install -e ".[dev]"
```

如果你选择的是 `.venv_jax081`，建议显式固定 TPU JAX 版本：

```bash
pip uninstall -y jax jaxlib
pip install "jax[tpu]==0.8.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
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

说明当前是 CPU 后端（`jax.device_count()==1`）。在 TPU VM 上执行（按你选的环境二选一）：

```bash
# 如果当前是 .venv_jax081（保持 0.8.1）
pip uninstall -y jax jaxlib
pip install "jax[tpu]==0.8.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 如果当前是 .venv（不固定版本）
pip uninstall -y jax jaxlib
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

验证：

```bash
python -c "import jax; print('backend:', jax.default_backend()); print('device_count:', jax.device_count()); print('devices:', jax.devices())"
```

期望看到 `backend: tpu` 且 `device_count: 4`（或与你的 TPU 拓扑一致）。

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
