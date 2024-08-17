# 起源与发展
2017 年 Google 在[《Attention Is All You Need》](https://arxiv.org/abs/1706.03762)中提出了 Transformer 结构用于序列标注，在翻译任务上超过了之前最优秀的循环神经网络模型；与此同时，Fast AI 在[《Universal Language Model Fine-tuning for Text Classification》](https://arxiv.org/abs/1801.06146)中提出了一种名为 ULMFiT 的迁移学习方法，将在大规模数据上预训练好的 LSTM 模型迁移用于文本分类，只用很少的标注数据就达到了最佳性能。

这些具有开创性的工作促成了两个著名 Transformer 模型的出现：

[GPT](https://openai.com/blog/language-unsupervised/) (the Generative Pretrained Transformer)；

[BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers)。

通过将 Transformer 结构与无监督学习相结合，我们不再需要对每一个任务都从头开始训练模型，并且几乎在所有 NLP 任务上都远远超过先前的最强基准。

虽然新的 Transformer 模型层出不穷，它们采用不同的预训练目标在不同的数据集上进行训练，但是依然可以按模型结构将它们大致分为三类：

- 纯 Encoder 模型（例如 BERT），又称自编码 (auto-encoding) Transformer 模型；
- 纯 Decoder 模型（例如 GPT），又称自回归 (auto-regressive) Transformer 模型；
- Encoder-Decoder 模型（例如 BART、T5），又称 Seq2Seq (sequence-to-sequence) Transformer 模型。

# 什么是 Transformer

## 语言模型
`Transformer` 模型本质上都是预训练语言模型，大都采用自监督学习 (Self-supervised learning) 的方式在大量生语料上进行训练，也就是说，训练这些 `Transformer` 模型完全不需要人工标注数据。

例如下面两个常用的预训练任务：

- 基于句子的前 `n`个词来预测下一个词，因为输出依赖于过去和当前的输入，因此该任务被称为**因果语言建模** (causal language modeling)；
![](https://transformers.run/assets/img/transformers/causal_modeling.svg)
- 基于上下文（周围的词语）来预测句子中被遮盖掉的词语 (masked word)，因此该任务被称为**遮盖语言建模** (masked language modeling)。
![](https://transformers.run/assets/img/transformers/masked_modeling.svg)

这些语言模型虽然可以对训练过的语言产生统计意义上的理解，例如可以根据上下文预测被遮盖掉的词语，但是如果直接拿来完成特定任务，效果往往并不好。

因此，我们通常还会采用迁移学习 (transfer learning) 方法，使用特定任务的标注语料，以有监督学习的方式对预训练模型参数进行微调 (fine-tune)，以取得更好的性能。

## 迁移学习
预训练是一种从头开始训练模型的方式：所有的模型权重都被随机初始化，然后在没有任何先验知识的情况下开始训练。
这个过程不仅需要海量的训练数据，而且时间和经济成本都非常高。

因此，大部分情况下，我们都不会从头训练模型，而是将别人预训练好的模型权重通过迁移学习应用到自己的模型中，即使用自己的任务语料对模型进行“二次训练”，通过微调参数使模型适用于新任务。

这种迁移学习的好处是：

- 预训练时模型很可能已经见过与我们任务类似的数据集，通过微调可以激发出模型在预训练过程中获得的知识，将基于海量数据获得的统计理解能力应用于我们的任务；
- 由于模型已经在大量数据上进行过预训练，微调时只需要很少的数据量就可以达到不错的性能；
- 换句话说，在自己任务上获得优秀性能所需的时间和计算成本都可以很小。

例如，我们可以选择一个在大规模英文语料上预训练好的模型，使用 `arXiv` 语料进行微调，以生成一个面向学术/研究领域的模型。这个微调的过程只需要很少的数据：我们相当于将预训练模型已经获得的知识“迁移”到了新的领域，因此被称为**迁移学习**。

与从头训练相比，微调模型所需的时间、数据、经济和环境成本都要低得多，并且与完整的预训练相比，微调训练的约束更少，因此迭代尝试不同的微调方案也更快、更容易。实践证明，即使是对于自定义任务，除非你有大量的语料，否则相比训练一个专门的模型，基于预训练模型进行微调会是一个更好的选择。

**在绝大部分情况下，我们都应该尝试找到一个尽可能接近我们任务的预训练模型，然后微调它**，也就是所谓的“站在巨人的肩膀上”。

## Transformer 的结构
标准的 Transformer 模型主要由两个模块构成：
![](https://transformers.run/assets/img/transformers/transformers_blocks.svg)
Encoder（左边）：负责理解输入文本，为每个输入构造对应的语义表示（语义特征）；
Decoder（右边）：负责生成输出，使用 Encoder 输出的语义表示结合其他输入来生成目标序列。

这两个模块可以根据任务的需求而单独使用：

- 纯 Encoder 模型：适用于只需要理解输入语义的任务，例如句子分类、命名实体识别；
- 纯 Decoder 模型：适用于生成式任务，例如文本生成；
- Encoder-Decoder 模型或 Seq2Seq 模型：适用于需要基于输入的生成式任务，例如翻译、摘要。


