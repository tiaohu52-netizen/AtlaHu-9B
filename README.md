# AtlaHu-9B

AtlaHu 是我基于商汤 SenseNova-Loop-9B 制作的量化与推理优化版本。我希望在单张消费级显卡上保留原模型的双遍循环，同时少占显存、缩短逐 token 解码的等待时间。我使用自研工具 Atlas-RSI 定位耗时和占用显存的模块，编辑已有参数、校准量化精度，并优化循环的执行方式。整个过程沿用原模型权重，没有重新预训练、指令微调或强化学习后训练。[商汤原模型](https://huggingface.co/sensenova/SenseNova-Loop-9B) · [上游项目](https://github.com/OpenSenseNova/SenseNova-Loop)

在同一张 **RTX 4080 SUPER** 上，与官方 BF16 权重通过 Transformers 参考实现运行的结果相比，AtlaHu 的双遍缓存解码吞吐从 **1.90 提至 15.47 token/s（8.15 倍）**，模型加载后的显存分配从 **18.84 降至 10.70 GB（减少 43.23%）**。四个固定前缀的平均负对数似然（NLL）从 1.3497 降至 1.3237。[逐轮测量记录](benchmarks/paired.json) · [汇总](benchmarks/summary.json)

## 同卡对照

| 指标 | 官方 BF16 | AtlaHu-9B |
| --- | ---: | ---: |
| 双遍缓存解码吞吐 | 1.899 token/s | 15.469 token/s |
| 输入预填中位耗时 | 576.6 ms | 288.7 ms |
| 模型加载后的 CUDA 分配增量 | 18.844 GB | 10.698 GB |
| 主推理张量文件体积 | 18.832 GB | 10.649 GB |
| 固定参考 token 平均 NLL，越低越好 | 1.349695 | 1.323748 |

批量为 1。四个评测前缀分别有 95、88、68、74 个 token，每个前缀测 6 轮，两版交替先运行。每轮独立预填并预热 3 步，再计时 16 个固定参考 token。两版各测 384 个解码位置，总耗时分别为 202.261 秒和 24.824 秒。显存指模型加载时的分配增量，GB 按十进制计算。官方侧使用上游 BF16 权重与模型代码，环境中未安装可选的 FLA 和 causal-conv1d，使用了 Transformers 的回退算子。8.15 倍是这两套配置在本机的实测比值。[环境与文件口径](benchmarks/environment.json)

质量评测使用 80 个不同的参考 token。AtlaHu 相对官方的平均 NLL 变化为 **−0.02595 nat/token**。分来源看，代码为 +0.0627，逻辑为 −0.0823，数学推理为 −0.1551，数学问答为 +0.0709。这组数据支持“小样本平均 NLL 略有改善”，还不足以推断所有任务的质量或长文本稳定性。[分来源数据](benchmarks/summary.json)

## Atlas-RSI 改了哪些地方

- **循环和参数。** 保留 32 个物理文本层，中间 24 层各运行两遍，一次前向共访问 56 层。调整了部分前馈网络通道，并根据固定参考 token 的评测选择两遍循环之间的三个 BF16 残差交接系数。
- **权重和缓存。** 249 个线性权重矩阵使用 INT8 与对应的缩放系数，输入嵌入按 32 个元素分组量化为 INT8。全注意力的 KV 缓存也采用 INT8。
- **线性注意力。** Atlas-RSI 定位到 24 个物理模块中 120 个矩阵的缩放系数存在大量重复。50,528,256 个 FP32 系数实际只有 990 种取值。用 10 位索引保存原值，比上一个紧凑版本再省 138,948,528 字节。只有这一步是无损重编码，所有系数都能精确还原。
- **执行方式。** 加载器挂载与该检查点配套的融合投影、递归状态更新和 CUDA Graph，减少逐 token 解码的调度开销。

## 下载和加载

Git 仓库包含分词器、配置、模型代码和推理加载器。四个 safetensors 文件随 Release 发布，合计 11,135,262,406 字节，其中独立的 MTP 权重为 486,582,880 字节。下载脚本逐卷核对 SHA-256，重组后再核对整文件。重复运行可从已校验的分卷继续。[权重下载页](https://github.com/tiaohu52-netizen/AtlaHu-9B/releases/tag/atlahu-v1.0.0) · [文件清单与校验值](weights.json)

```bash
git clone https://github.com/tiaohu52-netizen/AtlaHu-9B.git
cd AtlaHu-9B
python download_weights.py
```

实测环境为 Linux、Python 3.12、PyTorch 2.13.0+cu130、Transformers 5.19.0、Triton 3.7.1 和 safetensors 0.8.0。随附的优化配置会核对 GPU 计算能力与算子版本，已验证的计算能力为 8.9。请在仓库根目录用配套加载器打开权重。下面的示例以固定输入执行一次前向并打印输出张量形状。

```bash
PYTHONPATH=inference python - <<'PY'
import torch
from transformers import AutoTokenizer
from checkpoint import open_compact

tokenizer = AutoTokenizer.from_pretrained("AtlaHu", local_files_only=True)
inputs = tokenizer("你好", return_tensors="pt").to("cuda")
with open_compact("AtlaHu") as loaded, torch.inference_mode():
    result = loaded.model(**inputs, use_cache=True, logits_to_keep=1)
    print(tuple(result.logits.shape))
PY
```

## 来源与许可

AtlaHu-9B 是基于商汤 SenseNova-Loop-9B 的社区衍生版本。商汤与上游贡献者保有原模型和原始代码的署名。本仓库沿用 Apache License 2.0，AtlaHu 的参数编辑、量化存储和推理实现由本项目提供。[许可证](LICENSE) · [上游中文说明](README_CN.md)
