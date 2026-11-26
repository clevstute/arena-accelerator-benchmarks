# Arena 在天数智芯 MR-V100 上的训练测试

本包对应结题测试五「负载分割优化功能验证」。在单机双 MR-V100 上，Arena 选出的 `2/1/1 [4]` 在三个实测候选中最快。相对于手工均分流水 `1/2/1 [2,2]`，训练吞吐从 **3,360.41** 提升到 **3,662.19 tokens/s**（**+8.98%**），完整训练步耗时缩短 **8.24%**。

| 方案 | TP / PP / DP | 流水段层数 | 预测前后向（ms） | 实测前后向（ms） | 完整训练步（ms） | 吞吐（tokens/s） |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 手工均分流水 | 1 / 2 / 1 | [2, 2] | 128.11 | 534.23 | 609.45 | 3,360.41 |
| Arena 选案 | 2 / 1 / 1 | [4] | 106.12 | 470.24 | 559.23 | 3,662.19 |
| 自动流水分层 | 1 / 2 / 1 | [3, 1] | 106.68 | 516.78 | 615.95 | 3,324.96 |

三个布局使用同一个 4 层 FP16 GPT，隐藏维度为 4,096，注意力头数为 32，词表大小为 50,000。序列长度为 256，全局 batch size 为 8，每步包含 4 个 micro-batch，每个含 2 个样本。先对各布局做单步正确性检查，再以相反顺序各测一轮。每次运行预热 2 步、计时 5 步，两个 rank 的抽样权重均验证 7 次更新。完整训练步耗时取两 rank 的较大值后求平均，吞吐按每步 2,048 个 token 除以平均耗时计算。

对 TP=2 补测词表并行交叉熵的三次 FP32 规约，测得每个 micro-batch 约 0.303 ms，为前后向预测增加约 1.21 ms。Arena 在补计前后均选中 `2/1/1 [4]`。环境、验证范围及计时细节见[结题测试报告](reports/天数MR-V100结题测试报告.docx)。原轮次的数值及两轮吞吐见[控制台汇总摘录](results/summary-from-console.json)。

## 单机环境和输入源码

- 一台 Linux 主机使用两张 MR-V100，设备节点为 `/dev/iluvatar0`、`/dev/iluvatar1` 和 `/dev/itrctl`，训练使用 PyTorch 的 CUDA 设备接口和 NCCL 后端。
- 实测使用 CoreX 5.0.0 与 PyTorch 2.10.0 的 Ubuntu 24.04 容器，Python 版本为 3.12。脚本挂载宿主机 CoreX 驱动库，在容器里安装 `sentencepiece==0.2.1` 和 `pybind11==2.13.6` 并编译 Megatron 的数据集扩展。如果镜像缺少编译工具或 Python 头文件，准备阶段会通过 APT 安装。
- 输入为原实测的 `arena-iluvatar-src.tar.gz`，归档根目录包含 `crius-torch/` 与 `Megatron-LM/`。原输入由已准备训练适配的寒武纪工作树打包。上游资料见 [Arena 论文](https://arxiv.org/abs/2403.16125)、[Crius MLU 分支](https://github.com/DicardoX/crius-torch/tree/mlu_bench)和 [Megatron-LM](https://github.com/DicardoX/Megatron-LM)。

在仓库根目录复制样例，按主机情况设置源码包路径、驱动库路径和工作目录，再加载配置。

```bash
cp config/node.env.example ~/arena-mrv100-node.env
# 编辑 ~/arena-mrv100-node.env 后执行
source ~/arena-mrv100-node.env
docker pull "$ARENA_MR_IMAGE"
```

需要重建输入包时，使用原 `arena-mlu-train-smoke` 工作树，也可以按寒武纪包 README 中「准备上游代码和训练补丁」的步骤准备该工作树。已有 `arena-iluvatar-src.tar.gz` 可直接用于下节。

```bash
source_dir=/path/to/arena-mlu-train-smoke
tar --exclude-vcs --exclude=__pycache__ -C "$source_dir" \
  -czf "$ARENA_MR_SOURCE_ARCHIVE" crius-torch Megatron-LM
```

## 验证和测量

```bash
bash scripts/run-test5.sh
```

入口脚本在工作目录下创建新的试验子目录，解包源码、编译数据集扩展、应用词表并行损失修复。随后采集单机算子和双卡通信开销，补测词表通信，调用 Arena 评分并对候选执行正确性校验和训练计时。若要复用同一主机和负载的采样，可在配置中设置 `ARENA_MR_PROFILE_FILE`。

已有词表修复目录时，可直接调用四层对照脚本。

```bash
bash scripts/arena-iluvatar-test5-4layer-vocab-cost.sh /path/to/vocab-fixed-run
```

`TEST5_MR_4L_DIR` 后面是该次试验结果目录。`candidates.json` 和 `selection.json` 记录评分和选案，`vocab-communication.json` 保存词表通信采样，`summary.json` 汇总实测。汇总字段 `manual_layout` 对应手工均分流水 `pp2_uniform`，`arena_vs_manual_throughput_pct` 给出相对该对照的吞吐变化。各候选的验证日志和逐轮结果保存在同一目录中。
