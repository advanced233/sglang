# 运行环境配置清单（共享服务器）

本文档面向在共享服务器上执行配置的本地 agent，配合 [`plan.md`](plan.md) 使用。

- 只在 **A100-40G** 那台机器上执行。A100-80G 是共享卡，只用于可选项，等用户明确要求时再照本文另配一遍。
- 两张卡在同一台机器上时，执行一遍即可。

## 0. 共享服务器规则（全程遵守）

**写入范围**

- 所有文件只写到自己账号名下的工作根目录 `$KVQ_WORK`，默认是 `$HOME/kvq-work`。
- 工具在 `$HOME` 下自动生成的少量配置和缓存（例如 `~/.conda/`、`~/.cache/flashinfer`）也属于自己账号，可以接受。
- 如果 `$HOME` 配额不够，只能改用已经存在、且属主是自己的大盘目录（用 `stat -c %U <目录>` 确认）。不要在公共盘上新建目录。

**禁止的操作**

- 不用 `sudo`，不用 `apt` / `yum`，不用 Docker（需要 docker 组权限）。
- 不写 `/usr`、`/opt`、`/etc`，也不写公共数据盘或公共模型目录。
- 不修改系统级的 CUDA、驱动、conda base 环境和 `/etc/pip.conf`。

**Python 包隔离**

- 不用 `pip install --user`；设置 `PYTHONNOUSERSITE=1`，避免读到 `~/.local` 里的包。
- SGLang、vLLM、sgl-eval 各用一个独立环境。三者的 torch 版本和依赖不同，装在一起会互相覆盖。sgl-eval 必须在自己的环境里运行，这是仓库规则（`.claude/rules/prefer-sgl-eval.md`）。

**缓存与临时文件**

- pip、uv、conda、HuggingFace、Triton、inductor、CUDA、SGLang JIT、vLLM 的缓存以及 `TMPDIR`，全部指向 `$KVQ_WORK` 下（见步骤 2）。
- 系统 `/tmp` 是公用的，编译时尤其不要往里写。

**使用 GPU**

- 每次运行前先用 `nvidia-smi` 查看占用情况，用 `CUDA_VISIBLE_DEVICES` 只选一张空闲的卡。本项目所有实验都是单卡。
- 空闲卡的定义：
  - 这张卡上没有计算进程（`nvidia-smi --query-compute-apps` 里没有它）；
  - 显存占用低于 1 GB。
  - Xorg、gnome-shell 这类图形进程（`nvidia-smi` 进程表里类型为 `G`）只占几十 MB，不算占用。
- 不要 kill 别人的进程，不改 GPU 时钟、persistence mode、MIG 或 ECC。
- 如果服务器有**正在运行**的 Slurm 之类的调度器（`sinfo` 能列出分区），就通过调度器申请 GPU。只装了客户端、没有配置文件也没有服务进程，不算有调度器。
- 如果用沙箱（例如 bwrap）包装命令，GPU 相关的步骤必须把 `/dev/nvidia*`（含 `nvidiactl`、`nvidia-uvm`、`nvidia-uvm-tools`）透传进去，否则 CUDA 不可见。

**推理服务进程**

- SGLang 和 vLLM 的服务一律只监听 `127.0.0.1`，端口每次随机取一个高位端口，不用默认的 30000 / 8000。
- 启动时记下自己服务进程的 PID，结束时只按这个 PID 停止（先 `kill <PID>`，等 30 秒还在再 `kill -9`）。
  - 不要用 `killall_sglang`、`pkill -f sglang` 这类按名字匹配的命令，它们可能误杀别人的进程。
- 每次自检结束都要把自己起的服务停掉，不留常驻进程占卡。

**避免和他人冲突**

- 编译时限制并行度（`MAX_JOBS=4`），运行时限制 `OMP_NUM_THREADS`，避免占满共享 CPU。
- 不修改 `~/.bashrc`：环境变量写进 `$KVQ_WORK/env.sh`，需要时手动 `source`。

**源码与分支**

- 仓库是 `advanced233/sglang`，分支 `kv-int8-int4`。源码只由 Claude 修改，agent 不改、不提交、不推送。
- 本次配置的回报只写在本地（步骤 9），不进仓库：里面有主机名、用户名和路径。

**无人值守执行**：用户不在场，整个流程不需要等确认，从头跑到尾。遇到阻塞时按下表处理，绝不为了绕过阻塞而越过上面的规则：

| 情况 | 处理 |
|---|---|
| 某一步需要 root | 跳过该项，记录原因，继续其余步骤 |
| 驱动低于 580（不支持 CUDA 13） | 按步骤 4 尝试 CUDA 前向兼容包；仍然不行，就跳过 SGLang 的 GPU 自检，其余步骤（vLLM、模型下载、CPU 测试）照常执行，并在回报中写明。**不要**自行换成旧版 SGLang |
| GPU 全被占用 | 每 10 分钟重新检查一次，最多等 1 小时；仍然没有空闲卡就跳过 GPU 相关的自检，记录原因 |
| 某个源访问不了 | 尝试 `env.sh` 里注释掉的镜像选项；模型可以改从 ModelScope 下载（步骤 7）；仍然失败就跳过依赖它的项 |
| Llama-3.1-8B-Instruct 下载不了（gated，没有 token 或没有授权） | 先试 ModelScope；仍然不行就只下载 Qwen3-8B，记录原因。Llama 只用于交叉验证，不阻塞 |
| `ncu` 没有性能计数器权限 | 记录，不要尝试修改驱动参数 |
| 需要写入自己目录以外的位置 | 不写，跳过该项并记录 |
| 发现公共的模型副本 | 不使用，照常下载自己的副本，并在回报中提一句 |
| 某个包安装或 JIT 编译失败 | 按对应步骤里的退路处理；退路也失败就记录完整报错，继续后面的步骤 |

## 1. 清单

| # | 项 | 要求 / 目标 | 完成标志 |
|---|---|---|---|
| 1 | 工作根目录 | `$KVQ_WORK`，属主是自己；可用空间 ≥ 150 GB（三个环境约 45 GB，两个模型约 33 GB，加上缓存、输出和 profile） | `df -h $KVQ_WORK` |
| 2 | `env.sh` | 所有缓存、`TMPDIR`、`HF_HOME` 都指向 `$KVQ_WORK`；按需切换三个环境 | `source` 后 `env` 可见 |
| 3 | 驱动 | ≥ 580 可以直接使用 CUDA 13；更低时走步骤 4 的前向兼容 | `nvidia-smi` |
| 4 | conda | 自己目录下的 Miniforge；三个环境用 `-p` 建在 `$KVQ_WORK/envs/` 下，Python 3.12 | `which python` 指向对应目录 |
| 5 | SGLang 环境 | `envs/sglang`：分支 `kv-int8-int4` 的 editable 安装，`pip install -e "python[test]"`；torch 2.14.1（CUDA 13）；env 里有与之匹配的 nvcc 13 | `import sglang`；`torch.version.cuda` 以 13 开头；JIT kernel 能编译 |
| 6 | vLLM 环境 | `envs/vllm`：含 `int8_per_token_head` 和 `int4_per_token_head` 的版本 | 步骤 6 的检查脚本列出这两个 dtype |
| 7 | sgl-eval 环境 | `envs/sgleval`：`sgl-eval==0.1.3`（与 `python/pyproject.toml` 的 `test` extra 一致） | `sgl-eval --help` |
| 8 | Nsight Systems / Compute | 优先用系统已有的；没有就装进 SGLang 环境；记录 `ncu` 是否有计数器权限 | `nsys --version`、`ncu --version` |
| 9 | 模型权重 | `Qwen/Qwen3-8B`（必需），`meta-llama/Llama-3.1-8B-Instruct`（交叉验证，可缺） | 设置 `HF_HUB_OFFLINE=1` 后可以加载 |
| 10 | 仓库测试 | 两个与本项目直接相关的测试文件：一个 CPU、一个 GPU | 记录 passed / failed / skipped 数 |
| 11 | SGLang 冒烟 | Qwen3-8B 分别用 flashinfer、triton、triton + `fp8_e4m3`、triton + `fp4_mx_block16` 启动，各发一条请求 | 每组记录能否启动和 `max_total_num_tokens` |
| 12 | 压测与评测工具 | `bench_serving` 跑一次小规模压测；`sgl-eval` 跑 20 道 GSM8K | 有数字输出 |
| 13 | vLLM 冒烟 | Qwen3-8B 用 `TRITON_ATTN` 后端，分别开 `int8_per_token_head` 和 `int4_per_token_head`，各发一条请求 | 每组记录能否启动和 KV cache 容量 |
| 14 | 环境记录 | GPU 型号（PCIe / SXM）、时钟、温度、功率上限，以及所有版本号 | 写进回报 |

## 2. 执行步骤

### 步骤 1：只读探查（结果写进回报，不用等确认，直接继续）

```bash
whoami; echo "$HOME"; hostname
df -h "$HOME"; quota -s 2>/dev/null || true
# 是否已有属于自己的大盘目录（只看，不创建）
for d in /data/$USER /scratch/$USER /mnt/*/$USER; do [ -d "$d" ] && stat -c '%U %n' "$d"; done 2>/dev/null
nvidia-smi
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,power.limit,clocks.max.sm,clocks.max.mem --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
grep -i RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null   # 1 表示只有 root 能用 ncu 的性能计数器
which nvcc && nvcc --version; ls -d /usr/local/cuda* 2>/dev/null
gcc --version | head -1; ldd --version | head -1
free -g; nproc
which conda python3 uv 2>/dev/null; ls -d "$HOME"/xdit-work/miniforge3 2>/dev/null
which sbatch srun squeue sinfo 2>/dev/null
which nsys ncu 2>/dev/null; cat /proc/sys/kernel/perf_event_paranoid
curl -sI https://huggingface.co | head -1; curl -sI https://pypi.org/simple/ | head -1
curl -sI https://download.pytorch.org/whl/ | head -1; curl -sI https://github.com | head -1
curl -sI https://wheels.vllm.ai/nightly/ | head -1; curl -sI https://www.modelscope.cn | head -1
```

`$KVQ_WORK` 按以下规则自动决定：
- `$HOME` 所在盘的可用空间（并且配额允许）≥ 150 GB 时，用 `$HOME/kvq-work`。
- 否则，如果上面找到了属主是自己、空间足够的大盘目录，就用其下的 `kvq-work`。
- 两者都不满足时，仍用 `$HOME/kvq-work`，步骤 7 只下载 Qwen3-8B，并记录缺口。

记入回报的内容：
- 最终使用的 `$KVQ_WORK`、选择它的理由、可用空间。
- GPU 型号：要写清 `A100-PCIE-40GB` 还是 `A100-SXM4-40GB`，两者的功耗和散热不同。另外记录驱动版本、当前占用、最大时钟、功率上限。
- 驱动是否 ≥ 580。
- `RmProfilingAdminOnly` 的值。
- 是否有调度器，能否访问外网（HuggingFace、PyPI、PyTorch、GitHub、vLLM wheel 源、ModelScope）。
- 是否已有 `nsys` / `ncu`。

访问不了 huggingface.co 时，可以设置 `HF_ENDPOINT` 指向镜像，或改用 ModelScope（步骤 7）。pip 访问慢时，可以设置 `PIP_INDEX_URL`。这两项都只写进 `env.sh`，不改全局配置。

### 步骤 2：目录与 `env.sh`

```bash
export KVQ_WORK="$HOME/kvq-work"   # 或步骤 1 确认过的、属于自己的大盘目录
mkdir -p "$KVQ_WORK"/{envs,src,scripts,outputs,logs,profiles,tmp,cache,models}
cat > "$KVQ_WORK/env.sh" <<EOF
export KVQ_WORK="$KVQ_WORK"
EOF
cat >> "$KVQ_WORK/env.sh" <<'EOF'
export HF_HOME="$KVQ_WORK/cache/hf"
export MODELSCOPE_CACHE="$KVQ_WORK/cache/modelscope"
export PIP_CACHE_DIR="$KVQ_WORK/cache/pip"
export UV_CACHE_DIR="$KVQ_WORK/cache/uv"
export TORCH_HOME="$KVQ_WORK/cache/torch"
export TRITON_CACHE_DIR="$KVQ_WORK/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$KVQ_WORK/cache/inductor"
export CUDA_CACHE_PATH="$KVQ_WORK/cache/nv"
export SGLANG_CACHE_DIR="$KVQ_WORK/cache/sglang"
export SGLANG_JIT_CACHE_DIR="$KVQ_WORK/cache/sglang/jit"
export VLLM_CACHE_ROOT="$KVQ_WORK/cache/vllm"
export XDG_CACHE_HOME="$KVQ_WORK/cache/xdg"
export CONDA_PKGS_DIRS="$KVQ_WORK/cache/conda-pkgs"
export TMPDIR="$KVQ_WORK/tmp"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MAX_JOBS=4
# export HF_ENDPOINT=...        # 需要镜像时再打开
# export PIP_INDEX_URL=...
# 模型路径：默认用 HF repo id（走 $HF_HOME 缓存）；改从 ModelScope 下载时，改成本地目录（见步骤 7）
export QWEN3_8B="Qwen/Qwen3-8B"
export LLAMA31_8B="meta-llama/Llama-3.1-8B-Instruct"
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$SGLANG_JIT_CACHE_DIR" "$VLLM_CACHE_ROOT" \
         "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS" "$TMPDIR"
[ -f "$KVQ_WORK/miniforge3/bin/activate" ] && source "$KVQ_WORK/miniforge3/bin/activate"

# 切换环境：kvq_env sglang | vllm | sgleval
kvq_env() {
  conda activate "$KVQ_WORK/envs/$1" || return 1
  export CUDA_HOME="$CONDA_PREFIX"
  # 驱动低于 580 且步骤 4 启用了前向兼容时，这个文件才存在
  [ -f "$CONDA_PREFIX/kvq_compat.sh" ] && source "$CONDA_PREFIX/kvq_compat.sh"
  return 0
}
EOF
source "$KVQ_WORK/env.sh"
```

之后每次开新 shell，都先执行 `source $KVQ_WORK/env.sh`，再用 `kvq_env <名字>` 进入对应的环境。

### 步骤 3：conda 与三个环境

```bash
curl -L -o "$TMPDIR/miniforge.sh" \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash "$TMPDIR/miniforge.sh" -b -p "$KVQ_WORK/miniforge3"   # -b：不执行 conda init，不改 ~/.bashrc
source "$KVQ_WORK/env.sh"
for e in sglang vllm sgleval; do conda create -y -p "$KVQ_WORK/envs/$e" python=3.12; done
kvq_env sglang && which python pip   # 必须指向 $KVQ_WORK/envs/sglang/bin
```

- 如果步骤 1 发现了 Wan 项目留下的 `~/xdit-work/miniforge3`，也不要复用，照常在 `$KVQ_WORK` 下装一份。两个项目的环境要分开。
- 系统里已有的 conda，也不要使用它的 base 环境或公共环境。

### 步骤 4：SGLang 环境

SGLang 主线只支持 CUDA 13（`docs/docs/get-started/install.mdx` 写明 CUDA 12 的 wheel 已经停止发布），依赖里有 `torch==2.14.1`、`flashinfer_python[cu13]`、`cuda-python>=13`。

CUDA 13 需要 R580 及以上的驱动。CUDA 12 的"次版本兼容"（驱动 ≥ 525 即可运行任何 12.x 程序）不适用于跨大版本。

**4.1 安装**

```bash
source "$KVQ_WORK/env.sh" && kvq_env sglang
cd "$KVQ_WORK/src"
git clone --depth 50 -b kv-int8-int4 https://github.com/advanced233/sglang.git
cd sglang && git log -1 --format='%h %s'
pip install --upgrade pip
pip install -e "python[test]"
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 期望 2.14.1，CUDA 13.x
```

如果 pip 解析依赖很慢或失败，改用 uv：

```bash
pip install uv
uv pip install --prerelease=allow -e "python[test]"
```

**4.2 nvcc（SGLang 的 JIT kernel 运行时要编译）**

SGLang 的 JIT kernel 在运行时调用 nvcc 编译，按 `CUDA_HOME`、`PATH`、`/usr/local/cuda` 的顺序查找（`python/sglang/kernels/jit/utils/arch.py`）。所以要在 env 里装一个与 torch 的 CUDA 大版本和次版本都一致的 nvcc，不要用系统里版本不同的 CUDA。

```bash
CUDA_MM=$(python -c "import torch; print(torch.version.cuda)")   # 例如 13.0
conda install -y -c nvidia "cuda-nvcc=${CUDA_MM}" "cuda-cudart-dev=${CUDA_MM}" "cuda-cccl=${CUDA_MM}"
kvq_env sglang   # 重新进入，确保 CUDA_HOME=$CONDA_PREFIX
nvcc --version   # 应指向 $CONDA_PREFIX/bin/nvcc
# JIT 编译时如果报缺 CUDA 头文件，再补装：conda install -y -c nvidia "cuda-libraries-dev=${CUDA_MM}"
```

**4.3 GPU 可用性检查**

```bash
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
x = torch.ones(1024, device="cuda")
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
f = torch.compile(lambda t: (t * 2).sin())
print("triton/compile ok", f(x).sum().item())
print("float4_e2m1fn_x2", hasattr(torch, "float4_e2m1fn_x2"))   # fp4_mx_block16 需要它
EOF
```

**4.4 驱动低于 580 时：CUDA 前向兼容（只在 4.3 失败时做）**

A100 属于数据中心 GPU，支持 NVIDIA 的 CUDA 前向兼容：在旧的内核驱动之上，换用新版本的用户态驱动库（`libcuda.so`），整个过程不需要 root。能不能用取决于当前驱动所在的分支，以实测为准。

```bash
kvq_env sglang
CUDA_MM=$(python -c "import torch; print(torch.version.cuda)")
conda install -y -c conda-forge "cuda-compat=${CUDA_MM}" || conda install -y -c nvidia "cuda-compat=${CUDA_MM}"
COMPAT_DIR=$(dirname "$(find "$CONDA_PREFIX" -name 'libcuda.so*' -path '*compat*' | head -1)")
echo "$COMPAT_DIR"   # 应在 $CONDA_PREFIX 下
echo "export LD_LIBRARY_PATH=\"$COMPAT_DIR:\${LD_LIBRARY_PATH}\"" > "$CONDA_PREFIX/kvq_compat.sh"
kvq_env sglang
# 再执行一遍 4.3
```

- 4.3 通过，就在回报里写明"SGLang 环境启用了前向兼容"和 `COMPAT_DIR`。
- 仍然失败，就删除 `$CONDA_PREFIX/kvq_compat.sh`，跳过所有 SGLang 的 GPU 步骤（步骤 8.2、8.3、8.4 中的 SGLang 部分），记录完整报错，继续其他步骤。
- 不要升级驱动，也不要退回旧版 SGLang：本项目的代码基于主线 `a1966e1`，换版本由用户决定。

**4.5 Torch / Triton 报 PTX 或 toolchain 错误时**

1. 设置 `TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"`（4.2 装的 nvcc 自带 ptxas），然后重试。
2. 仍然失败，记录完整报错。

### 步骤 5：Nsight Systems / Nsight Compute

- 系统已有 `nsys` / `ncu` 就直接用，记录版本。
- 没有的话，装进 SGLang 环境：先用 `conda search -c nvidia nsight` 查实际的包名和版本（例如 `nsight-compute`、`cuda-nsight-compute`、`nsight-systems`），选与 `${CUDA_MM}` 匹配的版本安装。装不上就记录，不要用 root 安装。
- `/proc/sys/kernel/perf_event_paranoid` ≥ 2 时 CPU 采样不可用，改用 `nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none`，GPU 时间线不受影响。
- `ncu` 需要性能计数器权限。`RmProfilingAdminOnly=1` 时普通用户会报 `ERR_NVGPUCTRPERM`。这种情况只做记录：项目里的带宽利用率改用"字节数 ÷ 冷 L2 计时"计算，不依赖 `ncu`。

### 步骤 6：vLLM 环境（参考实现，只用于对照实验）

需要的版本必须包含 `int8_per_token_head` 和 `int4_per_token_head`。参考 commit 是 `9bc1c8b4fede8d009b308fecf74e560a1b4f7eca`（2026-10-28 的主线）。

```bash
source "$KVQ_WORK/env.sh" && kvq_env vllm
pip install --upgrade pip uv
# 先装最新的正式版；uv 根据驱动自动选择 torch 的 CUDA 版本
uv pip install vllm --torch-backend=auto
vllm download-kernels
```

检查需要的两个 dtype 是否都在：

```bash
python - <<'EOF'
from typing import get_args
import vllm
from vllm.config.cache import CacheDType
print("vllm", vllm.__version__)
print([d for d in get_args(CacheDType) if d.endswith("per_token_head")])
EOF
```

如果输出里缺少 `int8_per_token_head` 或 `int4_per_token_head`，就改装参考 commit 的 nightly wheel：

```bash
export VLLM_COMMIT=9bc1c8b4fede8d009b308fecf74e560a1b4f7eca
uv pip install -U vllm --torch-backend=auto --extra-index-url "https://wheels.vllm.ai/${VLLM_COMMIT}"
# 驱动 ≥ 580 时可以指定 CUDA 13 变体：--extra-index-url "https://wheels.vllm.ai/${VLLM_COMMIT}/cu130"
vllm download-kernels
# 再跑一次上面的检查
```

- 记录最终的 vLLM 版本，以及 torch 的 CUDA 版本。
- vLLM 用的是 CUDA 12.x 还是 13.x 都可以：两边对照的都是 Triton kernel。但必须写进回报，因为 M5 的对照表要注明。
- 这个环境不需要 nvcc，也不需要前向兼容。

**sgl-eval 环境**（版本与 `python/pyproject.toml` 的 `test` extra 一致）：

```bash
source "$KVQ_WORK/env.sh" && kvq_env sgleval
pip install "sgl-eval==0.1.3"
sgl-eval --help | head -20
```

### 步骤 7：模型权重

```bash
source "$KVQ_WORK/env.sh" && kvq_env sglang
python - <<'EOF'
from huggingface_hub import HfApi
for repo in ["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct"]:
    try:
        info = HfApi().model_info(repo, files_metadata=True)
        print(repo, f"{sum(s.size or 0 for s in info.siblings) / 1e9:.1f} GB")
    except Exception as e:
        print(repo, "UNAVAILABLE", type(e).__name__, e)
EOF
df -h "$KVQ_WORK"

# 下载到 $HF_HOME/hub，不加 --local-dir；只要 safetensors 和配置文件
hf download Qwen/Qwen3-8B --exclude "*.pth" "*.bin" "original/*"   # 旧版 CLI 用 huggingface-cli download
hf download meta-llama/Llama-3.1-8B-Instruct --exclude "*.pth" "*.bin" "original/*"   # gated：需要已授权的 HF token（HF_TOKEN）
```

HuggingFace 不通，或者 Llama 没有授权时，改从 ModelScope 下载到 `$KVQ_WORK/models/`：

```bash
modelscope download --model Qwen/Qwen3-8B --local_dir "$KVQ_WORK/models/Qwen3-8B"
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --local_dir "$KVQ_WORK/models/Llama-3.1-8B-Instruct"
```

- 改用 ModelScope 后，把 `env.sh` 里的 `QWEN3_8B` / `LLAMA31_8B` 改成对应的本地目录。后面所有命令都通过这两个变量引用模型。
- 下载完成后，运行时一律设置 `export HF_HUB_OFFLINE=1`。

### 步骤 8：自检

先选一张空闲的卡，再生成随机端口：

```bash
source "$KVQ_WORK/env.sh"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv   # 重新确认占用
export CUDA_VISIBLE_DEVICES=0   # 只选当前空闲的那张卡；没有空闲卡时按第 0 节的规则等待或跳过
export HF_HUB_OFFLINE=1
PORT=$((20000 + RANDOM % 20000))
```

启动服务和等待就绪用同一个辅助函数。脚本放在 `$KVQ_WORK/scripts`，不进仓库：

```bash
cat > "$KVQ_WORK/scripts/serve_lib.sh" <<'EOF'
# 用法：start_server <日志文件> <命令...>；设置 SERVER_PID
start_server() {
  local log="$1"; shift
  "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  echo "server pid $SERVER_PID, log $log"
}
# 用法：wait_ready <端口> <超时秒数>
wait_ready() {
  local port="$1" timeout="$2" t=0
  until curl -sf "http://127.0.0.1:${port}/v1/models" > /dev/null; do
    sleep 5; t=$((t + 5))
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server exited"; return 1; fi
    if [ "$t" -ge "$timeout" ]; then echo "timeout"; return 1; fi
  done
  echo "ready after ${t}s"
}
# 用法：stop_server；只停自己记下的 PID
stop_server() {
  kill "$SERVER_PID" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$SERVER_PID" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$SERVER_PID" 2>/dev/null
}
# 用法：ask <端口> <模型>；发一条非思考模式的短请求
ask() {
  curl -s "http://127.0.0.1:$1/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$2\", \"max_tokens\": 64, \"temperature\": 0,
    \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 * 23? Answer with the number only.\"}],
    \"chat_template_kwargs\": {\"enable_thinking\": false}}"
  echo
}
EOF
source "$KVQ_WORK/scripts/serve_lib.sh"
```

**8.1 仓库测试**

```bash
kvq_env sglang && cd "$KVQ_WORK/src/sglang"
python3 test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py     # CPU：KV 量化配方
python3 test/registered/attention/test_triton_attention_kernels.py                     # GPU：Triton decode / extend kernel
```

- 记录每个文件的 passed / failed / skipped 数。
- 这两个文件分别覆盖本项目要改的 KV 配方和 Triton attention kernel，后面会在它们上面加用例，所以必须先确认在这台机器上是绿的。
- 有失败就只记录现象，不要修改代码。

**8.2 SGLang 冒烟（Qwen3-8B）**

Qwen3 的 cookbook（`docs/cookbook/autoregressive/Qwen/Qwen3.mdx`）没有 A100 的条目；8B 推荐单卡 BF16，思考模式下加 `--reasoning-parser qwen3`。本项目的偏离点：
- 低比特 KV 只支持 triton 后端；
- 评测用非思考模式。

四组配置依次启动，每组发一条请求，然后停掉服务：

```bash
kvq_env sglang
for cfg in "default" "triton" "triton fp8_e4m3" "triton fp4_mx_block16"; do
  set -- $cfg
  args=(--model-path "$QWEN3_8B" --host 127.0.0.1 --port "$PORT")
  [ "$1" = "triton" ] && args+=(--attention-backend triton)
  [ -n "$2" ] && args+=(--kv-cache-dtype "$2")
  log="$KVQ_WORK/logs/smoke_sglang_${cfg// /_}.log"
  start_server "$log" python -m sglang.launch_server "${args[@]}"
  if wait_ready "$PORT" 900; then ask "$PORT" "$QWEN3_8B"; fi
  grep -o "max_total_num_tokens=[0-9]*" "$log" | tail -1
  grep -o "Use [a-z_0-9]* backend by default" "$log" | tail -1
  stop_server
done
```

- 记录每组：能否启动、启动耗时、`max_total_num_tokens`、实际用的 attention 后端、回答是否正确（391）。
- `fp4_mx_block16` 起不来（例如缺 `torch.float4_e2m1fn_x2`），就记录报错，这是 M0 要用的基线，但它不阻塞后面的步骤。

**8.3 压测与评测工具**

只验证工具能跑通，数字不作为结果：

```bash
kvq_env sglang
start_server "$KVQ_WORK/logs/smoke_bench.log" python -m sglang.launch_server --model-path "$QWEN3_8B" \
  --attention-backend triton --host 127.0.0.1 --port "$PORT"
wait_ready "$PORT" 900
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$PORT" \
  --dataset-name random --random-input-len 1024 --random-output-len 128 --random-range-ratio 1 \
  --num-prompts 32 --max-concurrency 8

# sgl-eval 在自己的环境里运行；服务保持开着
kvq_env sgleval
sgl-eval run --help | head -50   # 核对 --chat-template-kwarg 和采样参数的写法
sgl-eval run gsm8k --base-url "http://127.0.0.1:${PORT}/v1" --num-examples 20 \
  --chat-template-kwarg enable_thinking=false   # 参数写法以 --help 为准
kvq_env sglang && stop_server
```

- 记录 `bench_serving` 的输出吞吐、TPOT 中位数，以及 sgl-eval 的准确率和它打印的生成参数。
- 如果 `sglang.bench_serving` 的某个参数名不对，以 `python -m sglang.bench_serving --help` 为准，并在回报里写上实际用的命令。

**8.4 vLLM 冒烟**

```bash
kvq_env vllm
for dtype in int8_per_token_head int4_per_token_head; do
  log="$KVQ_WORK/logs/smoke_vllm_${dtype}.log"
  start_server "$log" vllm serve "$QWEN3_8B" --host 127.0.0.1 --port "$PORT" \
    --attention-backend TRITON_ATTN --kv-cache-dtype "$dtype"
  if wait_ready "$PORT" 900; then ask "$PORT" "$QWEN3_8B"; fi
  grep -i "KV cache size" "$log" | tail -1
  stop_server
done
```

- 记录每组：能否启动、KV cache 的 token 数、回答是否正确。
- 再起一组不带 `--kv-cache-dtype`（BF16）的作为容量参照。

### 步骤 9：回报

把下面的内容写到 `$KVQ_WORK/env_report.md`，用户回来后展示给他们看。主机名、用户名和路径不要提交到仓库。

```text
机器：GPU 型号（PCIe/SXM）| 驱动版本，是否 ≥ 580 | 最大 SM/显存时钟、功率上限、当前温度 | 是否有调度器
工作目录：$KVQ_WORK | 可用空间 | 已占用空间
SGLang 环境：commit | python / torch(cuda) / triton / flashinfer / sglang-kernel / transformers | nvcc 版本 | 是否启用前向兼容
vLLM 环境：vllm 版本（是否为 nightly，commit）| torch(cuda) | 两个 per_token_head dtype 是否都在
sgl-eval 环境：版本
工具：nsys / ncu 版本；RmProfilingAdminOnly；ncu 能否采集计数器
模型：Qwen3-8B / Llama-3.1-8B-Instruct 的来源（HF 或 ModelScope）、本地路径、大小
测试：8.1 两个文件各自的 passed / failed / skipped（附失败用例名）
SGLang 冒烟：四组配置各自能否启动、max_total_num_tokens、实际后端、回答
工具验证：bench_serving 的吞吐与 TPOT；sgl-eval 20 题准确率与生成参数
vLLM 冒烟：BF16 / INT8 / INT4 三组各自能否启动、KV cache 的 token 数、回答
未完成项和原因
按第 0 节规则自行做的决定（例如 $KVQ_WORK 的选择、前向兼容、跳过的项、等待 GPU 的情况）
```
