# 注意力机制
Transformer 模型之所以如此强大，是因为它抛弃了之前广泛采用的循环网络和卷积网络，而采用了一种特殊的结构——注意力机制 (Attention) 来建模文本。

本章将向大家介绍目前最常见的 `Multi-Head Attention`，并使用 `Pytorch` 框架实现一个 `Transformer block`。

## Attention
NLP 神经网络模型的本质就是对输入文本进行编码，常规的做法是首先对句子进行分词，然后将每个词语 (token) 都转化为对应的词向量 (token embeddings)，这样文本就转换为一个由词语向量组成的矩阵 $\boldsymbol{X}=(\boldsymbol{x}_1,\boldsymbol{x}_2,\dots,\boldsymbol{x}_n)$，其中 $\boldsymbol{x}_i$就表示第 $i$ 个词语的词向量，维度为 $d$，故 $\boldsymbol{X}\in \mathbb{R}^{n\times d}$ 
。

在Transformer 模型提出之前，对 token 序列 $X$ 的常规编码方式是通过循环网络 (RNNs) 和卷积网络 (CNNs)。
- RNN 的序列建模方式虽然与人类阅读类似，但是递归的结构导致其无法并行计算，因此速度较慢。而且 RNN 本质是一个马尔科夫决策过程，难以学习到全局的结构信息；  
- CNN 则通过滑动窗口基于局部上下文来编码文本，例如核尺寸为 3 的卷积操作就是使用每一个词自身以及前一个和后一个词来生成嵌入式表示。CNN 能够并行地计算，因此速度很快，但是由于是通过窗口来进行编码，所以更侧重于捕获局部信息，难以建模长距离的语义依赖。

Google《Attention is All You Need》提供了第三个方案：**直接使用 Attention 机制编码整个文本**。相比 RNN 要逐步递归才能获得全局信息（因此一般使用双向 RNN），而 CNN 实际只能获取局部信息，需要通过层叠来增大感受野，Attention 机制一步到位获取了全局信息：

$$\boldsymbol{y}_t = f(\boldsymbol{x}_t,\boldsymbol{A},\boldsymbol{B})$$

其中 $\boldsymbol{A}$， $\boldsymbol{B}$ 是另外的词语序列（矩阵），如果取 $\boldsymbol{A}=\boldsymbol{B}=\boldsymbol{X}$ 就称为 Self-Attention，即直接将 $\boldsymbol{x}_t$ 与自身序列中的每个词语进行比较，最后算出 $\boldsymbol{y}_t$。

### Scaled Dot-product Attention
Attention 有许多种实现方式，但是最常见的还是 Scaled Dot-product Attention。
<center>

![](https://transformers.run/assets/img/attention/attention.png)

</center>

Scaled Dot-product Attention 共包含 2 个主要步骤：

- **计算注意力权重**：使用某种相似度函数度量每一个 query 向量和所有 key 向量之间的关联程度。对于长度为 $m$ 的 Query 序列和长度为 $n$ 的 Key 序列，该步骤会生成一个尺寸为 $m\times n$ 的注意力分数矩阵。

    特别地，Scaled Dot-product Attention 使用点积作为相似度函数，这样相似的 queries 和 keys 会具有较大的点积。

    由于点积可以产生任意大的数字，这会破坏训练过程的稳定性。因此注意力分数还需要乘以一个缩放因子来标准化它们的方差，然后用一个 softmax 标准化。这样就得到了最终的注意力权重 $w_{ij}$，表示第 $i$ 个 query 向量与第 $j$ 个 key 向量之间的关联程度。

- **更新 token embeddings**：将权重 $w_{ij}$ 与对应的 value 向量 $\boldsymbol{v}_1,…,\boldsymbol{v}_n$ 相乘以获得第 $i$ 个 query 向量更新后的语义表示 $\boldsymbol{x}_i’ = \sum_{j} w_{ij}\boldsymbol{v}_j $。 

形式化表示为：

$$\text{Attention}(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V} \tag{4}$$

其中 $\boldsymbol{Q}\in\mathbb{R}^{m\times d_k}, \boldsymbol{K}\in\mathbb{R}^{n\times d_k}, \boldsymbol{V}\in\mathbb{R}^{n\times d_v}$ 分别是 query、key、value 向量序列。如果忽略 softmax 激活函数，实际上它就是三个 $m\times d_k,d_k\times n, n\times d_v$ 矩阵相乘，得到一个 $m\times d_v$ 的矩阵，也就是将 $m\times d_k$ 的序列 
 编码成了一个新的 $m\times d_v$ 的序列。

将上面的公式拆开来看更加清楚：
$$\text{Attention}(\boldsymbol{q}_t,\boldsymbol{K},\boldsymbol{V}) = \sum_{s=1}^n \frac{1}{Z}\exp\left(\frac{\langle\boldsymbol{q}_t, \boldsymbol{k}_s\rangle}{\sqrt{d_k}}\right)\boldsymbol{v}_s \tag{5}$$

其中 $Z$ 是归一化因子，$\boldsymbol{K},\boldsymbol{V}$ 是一一对应的 key 和 value 向量序列，Scaled Dot-product Attention 就是通过 $\boldsymbol{q}_t$ 这个 query 与各个 $\boldsymbol{k}_s$ 内积并 softmax 的方式来得到 $\boldsymbol{q}_t$ 与各个 $\boldsymbol{v}_s$ 的相似度，然后加权求和，得到一个 $d_v$ 维的向量。其中因子 $\sqrt{d_k}$ 起到调节作用，使得内积不至于太大。

下面我们通过 Pytorch 来手工实现 Scaled Dot-product Attention：

首先需要将文本分词为词语 (token) 序列，然后将每一个词语转换为对应的词向量 (embedding)。Pytorch 提供了 `torch.nn.Embedding` 层来完成该操作，即构建一个从 token ID 到 token embedding 的映射表：

```python
from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "time flies like an arrow"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(inputs.input_ids)

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_emb)

inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())
```

```python
tensor([[ 2051, 10029,  2066,  2019,  8612]])
Embedding(30522, 768)
torch.Size([1, 5, 768])
```

    为了演示方便，这里我们通过设置 `add_special_tokens=False` 去除了分词结果中的 `[CLS]` 和 `[SEP]`。

可以看到，BERT-base-uncased 模型对应的词表大小为 30522，每个词语的词向量维度为 768。Embedding 层把输入的词语序列映射到了尺寸为 `[batch_size, seq_len, hidden_dim]` 的张量。

接下来就是创建 query、key、value 向量序列 $\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$，并且使用点积作为相似度函数来计算注意力分数：

```python
import torch
from math import sqrt

Q = K = V = inputs_embeds
dim_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1,2)) / sqrt(dim_k)
print(scores.size())
```
```
torch.Size([1, 5, 5])
```

这里 $\boldsymbol{Q},\boldsymbol{K}$ 的序列长度都为 5，因此生成了一个 $5\times 5$ 的注意力分数矩阵，接下来就是应用 Softmax 标准化注意力权重：
```python
import torch.nn.functional as F

weights = F.softmax(scores, dim=-1)
print(weights.sum(dim=-1))
```
```
tensor([[1., 1., 1., 1., 1.]], grad_fn=<SumBackward1>)
```

最后将注意力权重与 value 序列相乘：
```python
attn_outputs = torch.bmm(weights, V)
print(attn_outputs.shape)
```
```
torch.Size([1, 5, 768])
```

至此就实现了一个简化版的 Scaled Dot-product Attention。可以将上面这些操作封装为函数以方便后续调用：
```python
import torch
import torch.nn.functional as F
from math import sqrt

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)
```

上面的代码还考虑了 $\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$ 序列的 Mask。填充 (padding) 字符不应该参与计算，因此将对应的注意力分数设置为 $-\infty$，这样 softmax 之后其对应的注意力权重就为 0 了（ $e^{-\infty}=0$ ）。

注意！上面的做法会带来一个问题：当 $\boldsymbol{Q}$ 和 $\boldsymbol{K}$ 序列相同时，注意力机制会为上下文中的相同单词分配非常大的分数（点积为 1），而在实践中，相关词往往比相同词更重要。例如对于上面的例子，只有关注“time”和“arrow”才能够确认“flies”的含义。

因此，多头注意力 (Multi-head Attention) 出现了！
