# Arena 在昇腾 910B3 上的训练测试

本包对应「异构计算平台的资源实时隔离与高效调度系统」结题测试五。单机双卡运行 4 层 FP16 GPT。原 Arena 按前后向预测选中自动流水 `1/2/1 [3,1]`。补入单独标定的完整步开销后，测试脚本改选均分流水 `1/2/1 [2,2]`。该布局在两轮独立复测中均最快，吞吐相对原选案提升 **5.75%**。

| 布局 | TP/PP/DP | 原始实测前后向（ms） | 原始实测完整步（ms） | 标定后预测完整步（ms） | 独立复测完整步（ms） | 独立复测吞吐（tokens/s） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 均分 PP2 [2,2] | 1/2/1 | 126.73 | 186.28 | **112.18** | **187.67** | **10,913.02** |
| 自动 PP2 [3,1] | 1/2/1 | **122.34** | 200.98 | 120.03 | 198.46 | 10,319.51 |
| TP2 [4] | 2/1/1 | 131.50 | 201.73 | 118.01 | 198.58 | 10,313.27 |

工作负载使用 MockGPTDataset，隐藏维度 4,096，注意力头数 32，词表大小 50,000。序列长度 256，全局 batch size 8，分为 4 个 micro-batch，每个含 2 个样本。各布局在计时前执行双卡单步校验。训练计时按相反顺序执行两轮，每轮预热 2 步、计时 5 步。每步取两 rank 的较长用时，吞吐按每步 2,048 个 token 除以平均完整步耗时计算。

原始选案使用本机算子与原生 HCCL 的采样数据。TP2 的词表并行通信每个 micro-batch 测得 0.513 ms，每步增加约 2.05 ms。完整步试验对每个布局另做短标定，取前后向窗口之外的步首与步尾平均开销，加入原评分。脚本先保存预测和选案，再运行两轮独立复测。完整步估时仍比复测低约 75–81 ms，这次验证覆盖同一负载下的逐布局标定与第一名预测。

[结题测试报告](reports/昇腾910B3结题测试报告.docx) · [原始控制台汇总](results/summary-from-console.json) · [完整步控制台汇总](results/full-step-summary-from-console.json)

## 实测环境与源码输入

- 一台 aarch64 Linux 主机使用两张昇腾 910B3，每张 HBM 64 GiB。实测宿主机运行 openEuler 22.03 LTS-SP4，`npu-smi` 版本为 25.5.0。
- 实测容器为 Ubuntu 22.04 ARM64，使用 CANN 8.2.RC1、Python 3.11、PyTorch 2.6.0+cpu 和 TorchNPU 2.6.0。两个 rank 使用原生 HCCL。
- 源码输入使用完成适配的 `arena-ascend-test5.tar.gz`。归档内含 `arena-ascend-test5/source/crius-torch/` 和 `arena-ascend-test5/source/Megatron-LM/`，以及昇腾 RNG 恢复、固定尺寸图导出和词表并行校验所需的修改。仓库中的 `workload/` 提供两套测试的启动脚本。
- 源码基准为 Crius `mlu_bench` 的 `f2a2aedeb9dd8e8be8aae073b5c7004a4aab7505` 和 Megatron-LM 的 `90cd925ed0842afd709b4bf63c56ee889f03323e`。

[Arena 论文](https://arxiv.org/abs/2403.16125) · [Crius MLU 分支](https://github.com/DicardoX/crius-torch/tree/mlu_bench) · [Megatron-LM](https://github.com/DicardoX/Megatron-LM)

## 容器准备

在有 Docker 和昇腾驱动的 Linux 主机上，从仓库根目录加载本机配置。把 `ARENA_ASCEND_SOURCE_ARCHIVE` 设为源码包的绝对路径，按分配的设备设置两个 NPU 编号。全新测试目录使用示例中的路径即可。

```bash
cp config/node.env.example ~/arena-ascend-node.env
# 编辑 ~/arena-ascend-node.env 后加载
source ~/arena-ascend-node.env
```

首次创建容器时，若镜像尚未缓存，先拉取配置中的 ARM64 镜像。

```bash
docker pull --platform linux/arm64 "$ARENA_ASCEND_IMAGE"
bash scripts/create-container.sh
bash scripts/install-workload.sh
```

安装脚本将源码解到容器中的工作目录，安装所需的 Python wheel，并在 `/opt/arena-venv` 里应用实测使用的 TorchNPU 2.6.0 导入兼容补丁。

已有通过验证的昇腾容器时，在配置里填写原容器名和原试验目录，然后同步 `workload/` 里的测试入口。

```bash
docker start "$ARENA_ASCEND_CONTAINER"
docker cp workload/. "$ARENA_ASCEND_CONTAINER:$ARENA_ASCEND_WORKLOAD_DIR/"
```

旧试验目录曾使用 `scripts/profile.py` 时，先将该文件移出 `scripts/`，以免和 Python 标准库的 `profile` 模块同名。

## 验证与计时

从 Linux 主机进入容器工作目录，依次运行单步验证、原始对照和完整步标定复测。

```bash
docker exec -it -w "$ARENA_ASCEND_WORKLOAD_DIR" "$ARENA_ASCEND_CONTAINER" bash
```

```bash
bash run-test5.sh verify
bash run-test5.sh run
bash run-full-step.sh
```

`run-full-step.sh` 默认读取最近一次完成的 `results/run.*` 目录，也可将已有试验目录作为参数。已有同机同负载的原始结果时，可在同步测试入口后直接启动完整步实验。

```bash
bash run-full-step.sh results/run.t0F79l
```

`TEST5_ASCEND_DIR` 指向原始结果目录，`summary.json` 记录前后向评分与训练对照。`TEST5_ASCEND_FULL_STEP_DIR` 指向完整步实验目录，其中 `full-step-prediction.json` 保存复测前的评分，`full-step-summary.json` 汇总预测、复测及选案是否命中。归档附带的 JSON 保留控制台的数值和字段，试验目录使用容器工作目录下的相对路径。
