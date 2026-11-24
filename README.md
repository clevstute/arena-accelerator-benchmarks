# Arena 在寒武纪 MLU580 上的训练测试

本分支对应结题测试五「负载分割优化功能验证」。两台主机各使用一张 MLU580，通过 CNCL SOCKET 跨机通信。对 4 层 FP16 GPT 使用相同负载，Arena 自动方案将完整训练步吞吐从 **713.42** 提升到 **3,940.90 tokens/s**，提升 **452.39%**。

| 方案 | TP / PP / DP | 流水段层数 | 完整训练步耗时 |
| --- | --- | --- | ---: |
| 手工基线 | 2 / 1 / 1 | [4] | 2,870.67 ms |
| Arena 自动方案 | 1 / 2 / 1 | [3, 1] | 519.68 ms |

全局 batch size 为 8，序列长度为 256，隐藏维度为 4,096，注意力头数为 32，词表大小为 50,000。每组交替重复 3 轮，每轮预热 2 步、计时 5 步。每个训练步取两个 rank 用时的最大值，吞吐按每步输入 2,048 个 token 除以平均完整训练步耗时计算，包含参数更新。

[结题测试报告](reports/寒武纪MLU580结题测试报告.docx)记录原验证环境、适配过程和计时方法。[控制台汇总摘录](results/summary-from-console.json)保留实测输出中的数值。机器生成的原始 `summary.json`、每轮日志和 tuner 记录保存在原试验节点，路径见报告。

## 上游版本和环境

- Crius 使用 `DicardoX/crius-torch` 的 `mlu_bench` 分支，基准提交为 `f2a2aedeb9dd8e8be8aae073b5c7004a4aab7505`。
- Megatron 使用 `DicardoX/Megatron-LM`，提交为 `90cd925ed0842afd709b4bf63c56ee889f03323e`，Megatron Core 版本为 `0.9.0rc0`。
- 实测软件为 Python 3.10.12、PyTorch 2.4.0+cpu、`torch_mlu` 1.22.2+torch2.4.0、NumPy 1.26.4。
- 实测驱动为 6.2.10，CNToolkit 为 3.15.7，CNRT 为 6.14.1，CNNL 为 1.28.6，CNCL 为 1.25.0。使用与目标驱动和 Python 匹配的 `torch_mlu` wheel。

上游资料：[Arena 论文](https://arxiv.org/abs/2403.16125) · [Crius MLU 分支](https://github.com/DicardoX/crius-torch/tree/mlu_bench) · [Megatron-LM](https://github.com/DicardoX/Megatron-LM) · [Cambricon torch_mlu](https://github.com/Cambricon/torch_mlu)

## 两节点配置

在两台 Linux 节点以相同版本检出本分支。各自复制 `config/node.env.example` 到仓库外，设置环境目录、节点 rank、rank 0 的内网地址和网卡。两台机器分别用 rank 0 和 rank 1，并填写同一个 `ARENA_MLU_MASTER_ADDR`。

```bash
cp config/node.env.example ~/arena-mlu-node.env
# 编辑 ~/arena-mlu-node.env 后，在启动脚本的 shell 中加载
source ~/arena-mlu-node.env
```

每个节点的 `ARENA_MLU_TRIAL_DIR` 下使用 `.venv`。跑通过的试验环境可直接设置目录并加载配置。新环境需先把配套的 PyTorch 和 `torch_mlu` 安装进该虚拟环境。

```bash
python3.10 -m venv "$ARENA_MLU_TRIAL_DIR/.venv"
source "$ARENA_MLU_TRIAL_DIR/.venv/bin/activate"
python -m pip install 'torch==2.4.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 /path/to/torch_mlu-1.22.2+torch2.4.0-cp310-cp310-linux_x86_64.whl
```

## 准备上游代码和训练补丁

如果从干净目录复现，在 rank 0 准备与原测试一致的上游源码和依赖。请从本分支仓库根目录执行。

```bash
bundle="$ARENA_MLU_TRIAL_DIR/arena-mlu-train-smoke"
mkdir -p "$bundle"
git clone https://github.com/DicardoX/crius-torch.git "$bundle/crius-torch"
git -C "$bundle/crius-torch" checkout --detach f2a2aedeb9dd8e8be8aae073b5c7004a4aab7505
git clone https://github.com/DicardoX/Megatron-LM.git "$bundle/Megatron-LM"
git -C "$bundle/Megatron-LM" checkout --detach 90cd925ed0842afd709b4bf63c56ee889f03323e
source "$ARENA_MLU_TRIAL_DIR/.venv/bin/activate"
python -m pip install setuptools==75.8.0 pybind11==2.13.6 wheel psutil packaging sentencepiece
CC=/usr/bin/gcc CXX=/usr/bin/g++ python -m pip install --no-build-isolation --no-deps "$bundle/Megatron-LM"
python -m pip install --no-build-isolation --no-deps -e "$bundle/crius-torch"
bash scripts/arena-mlu-init-fix.sh
bash scripts/arena-mlu-dual-smoke.sh pack
```

初始化脚本会补齐权重初始化、应用 MLU 优化器适配并跑 3 步单卡验证。打包步骤在 rank 0 的家目录生成 `arena-mlu-dual-input.tar.gz`。将它传到 rank 1 的家目录，再在 rank 1 加载配置并运行安装与检查。

```bash
bash scripts/arena-mlu-dual-smoke.sh install
```

## 验证和测量

两节点通过相同命令启动协作任务，先启动 rank 0 再启动 rank 1。每个阶段在两边结束后再继续。

| 步骤 | 节点 | 命令 |
| --- | --- | --- |
| 双机训练验证 | 两台 | `bash scripts/arena-mlu-dual-smoke.sh run` |
| 双机通信 profiling | 两台 | `bash scripts/arena-mlu-dual-tuner.sh comm` |
| 算子 profiling 和并行方案搜索 | rank 0 | `bash scripts/arena-mlu-dual-tuner.sh tune` |
| 手工与自动方案吞吐对照 | 两台 | `bash scripts/arena-mlu-dual-benchmark.sh` |

吞吐脚本重放本次已选出的 `1/2/1` 和 `[3,1]`，与手工 `2/1/1` 比较。对照生成的 `summary.json` 和各轮日志在 rank 0 启动时打印的结果目录中。仓库版本把节点名、地址、网卡和工作目录移入本机配置，沿用实测时的训练负载和计时逻辑。
