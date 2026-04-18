# Language-Modeling-from-Scratch
Comprehensive implementation of LLM components: BPE, Transformer, FlashAttention (Triton), and DPO. Inspired by Stanford CS336.

# 基于 Stanford CS336 的 Transformer 语言模型从零实现

这是一个参考 Stanford CS336 课程内容完成的从零实现小型语言模型项目。  
项目目标不是简单调用现成框架训练模型，而是尽可能自己实现一条完整的语言模型训练与推理流程，包括：

- BPE 分词与编码
- 数据预处理
- Decoder-only Transformer 搭建
- 因果自注意力机制
- RoPE 位置编码
- RMSNorm / 非 RMSNorm 两种结构
- AdamW 优化器
- 学习率预热与余弦退火
- 混合精度训练
- checkpoint 保存与断点续训
- 自回归文本生成推理

整个项目以 TinyStories 数据集为主要实验对象，重点放在**理解 Transformer 从数据到训练再到推理的完整实现过程**，以及在实际调试中解决各类工程问题。

---

## 1. Background

在学习 Stanford CS336 课程的过程中，我希望不仅停留在“知道 Transformer 的原理”，而是进一步理解它在代码层面到底如何真正跑起来。  
因此，这个项目尽量采用“自己搭建、自己排错、自己跑通”的方式，去复现一个小型语言模型的训练与推理流程。

和直接调用成熟框架相比，从零实现的过程虽然更慢，但能更清楚地理解很多细节问题，例如：

- tokenizer 的词表大小为什么必须和模型完全对齐
- attention mask 在混合精度下为什么可能溢出
- 为什么输出层 logits 会消耗大量显存
- checkpoint 为什么不能只保存模型参数
- 推理阶段为什么输入维度很容易出错
- 训练和推理的代码虽然看起来相似，但处理逻辑并不完全一样

---
## 2. Version
### **v1.0**

该decoder-only的transformer架构的整体框架已搭建完毕，主要模块如下所示：
1. 分词器与数据编码
   - 使用 BPE 思路处理文本
   - 构建 / 加载 `vocab.pkl` 和 `merges.pkl`
   - 将原始文本编码为 token id 序列
   - 将训练集和验证集预处理结果保存为 `.pkl` 文件，减少重复编码开销

2. Transformer 模型实现
   - 自行实现 decoder-only Transformer
   - 包含 embedding、Transformer block、前馈层、最终输出层
   - 支持因果 mask 的自注意力计算
   - 支持 RoPE 位置编码

3. 归一化与模型变体
   - 支持 RMSNorm 版本
   - 支持去掉 RMSNorm 的对照版本
   - 方便进行结构差异实验

4. 训练流程实现
   - 自定义训练循环
   - 使用 AdamW 优化器
   - 学习率 warmup + cosine schedule
   - 梯度裁剪
   - GPU 混合精度训练
   - 中途保存 checkpoint
   - 支持续训

5. 推理流程实现
   - 根据训练时一致的配置重建模型
   - 加载 checkpoint
   - 编码输入 prompt
   - 自回归生成文本
   - 使用 temperature scaling 和 top-p sampling 控制生成

---

##3. Architecture

下面的文件结构根据当前项目实际目录整理，和仓库内容保持一致：

```text
Language-Modeling-from-Scratch-based-on-Stanford-CS336
├── docs
├── src
│   ├── 1_tokenizer
│   │   ├── 1_1_tokenizer.py
│   │   └── 1_tokenizer.py
│   └── 2_transformer
│       ├── checkpoints
│       ├── pkls
│       ├── tmp_encoded_train_parts
│       ├── tmp_encoded_valid_parts
│       ├── __wandb
│       ├── wandb_logs
│       ├── adamw.py
│       ├── causal_multi_head_attention_no_weight.py
│       ├── cross_entropy.py
│       ├── dataloader.py
│       ├── embedding.py
│       ├── final_inference.py
│       ├── final_inference2.py
│       ├── final_train.py
│       ├── final_train2.py(- `final_train2.py`：当前主要使用的训练脚本，功能更完整，支持断点续训与验证集抽样验证`final_inference2.py`：当前主要使用的推理脚本，和最新训练配置保持一致 其余同类脚本为早期调试或过渡版本，保留用于过程记录)
│       ├── get_encoded_ids_train_valid.py
│       ├── inference.py
│       ├── lr_cosine_shedule.py
│       ├── pair_all_bpe_tokenzier.py
│       ├── RMSnorm.py
│       ├── rope.py
│       ├── SwiGLU.py
│       ├── temp.py
│       ├── tokenizer_encode.py
│       ├── train_script.py
│       ├── transformer_block_without_rmsnorm.py
│       ├── transformer_no_weight_block.py
│       ├── transformermodule.py
│       └── transformermodule_withoutrmsnorm.py
├── input
├── wandb
├── wandb_test.py
├── .gitignore
├── LICENSE
└── README.md
```



## 4. Main Files

为了方便理解整个工程，我把比较关键的文件简要说明如下：

### `src/1_tokenizer/`
这一部分主要是分词器相关代码，用于实现和测试 BPE tokenizer。

### `src/2_transformer/final_train2.py`
当前主要训练脚本。  
已经支持：

- GPU 训练
- 混合精度
- checkpoint 保存
- 验证集抽样验证
- 断点续训

### `src/2_transformer/final_inference2.py`
当前主要推理脚本。  
用于加载训练好的模型权重，输入 prompt 后生成文本。

### `src/2_transformer/transformermodule.py`
RMSNorm 版本的 Transformer 主体实现。

### `src/2_transformer/transformermodule_withoutrmsnorm.py`
不使用 RMSNorm 的模型版本，主要用于对照实验。

### `src/2_transformer/transformer_no_weight_block.py`
Transformer block 的核心实现，包括注意力层和前馈网络的组织方式。

### `src/2_transformer/causal_multi_head_attention_no_weight.py`
因果多头自注意力实现，是模型最核心的模块之一。

### `src/2_transformer/adamw.py`
自己实现的 AdamW 优化器。

### `src/2_transformer/lr_cosine_shedule.py`
学习率 warmup 与 cosine decay 调度器。

### `src/2_transformer/dataloader.py`
根据编码后的 token id 构造训练 batch 和验证 batch。

### `src/2_transformer/inference.py`
包含生成相关逻辑，例如 temperature scaling、top-p sampling、自回归 decode。

### `src/2_transformer/pkls/`
存放词表、merge 规则、训练集和验证集编码结果等数据文件。

### `src/2_transformer/checkpoints/`
存放训练过程中保存的中间 checkpoint 以及最终模型。

---

## 5. Config

当前训练脚本中使用的主要模型参数如下：

```python
vocab_size = 20000
context_length = 256
d_model = 512
d_ff = 1344
n_layers = 4
n_heads = 16
rope_theta = 10000.0
