# 参数高效微调（PEFT）方法
## 动机
基于 Transformers 架构的大型语言模型 (LLM)，如 GPT、T5 和 BERT，已经在各种自然语言处理 (NLP) 任务中取得了最先进的结果。此外，还开始涉足其他领域，例如计算机视觉 (CV) (VIT、Stable Diffusion、LayoutLM) 和音频 (Whisper、XLS-R)。传统的范式是对通用网络规模数据进行大规模预训练，然后对下游任务进行微调。与使用开箱即用的预训练 LLM (例如，零样本推理) 相比，在下游数据集上微调这些预训练 LLM 会带来巨大的性能提升。

然而，随着模型变得越来越大，在消费级硬件上对模型进行全部参数的微调变得不可行。此外，为每个下游任务独立存储和部署微调模型变得非常昂贵，因为微调模型与原始预训练模型的大小相同。参数高效微调(PEFT) 方法旨在解决这两个问题！

PEFT 方法仅微调少量 (额外) 模型参数，同时冻结预训练 LLM 的大部分参数，从而大大降低了计算和存储成本。这也克服了灾难性遗忘的问题，这是在 LLM 的全参数微调期间观察到的一种现象。PEFT 方法也显示出在低数据状态下比微调更好，可以更好地泛化到域外场景。它可以应用于各种模态，例如图像分类以及 Stable diffusion dreambooth。

PEFT 方法还有助于提高轻便性，其中用户可以使用 PEFT 方法调整模型，以获得与完全微调的大型检查点相比，大小仅几 MB 的微小检查点。例如， bigscience/mt0-xxl 占用 40GB 的存储空间，全参数微调将导致每个下游数据集有对应 40GB 检查点。而使用 PEFT 方法，每个下游数据集只占用几 MB 的存储空间，同时实现与全参数微调相当的性能。来自 PEFT 方法的少量训练权重被添加到预训练 LLM 顶层。因此，同一个 LLM 可以通过添加小的权重来用于多个任务，而无需替换整个模型。

简而言之，PEFT 方法使您能够获得与全参数微调相当的性能，同时只有少量可训练参数。

🤗 PEFT 库提供了最新的参数高效微调技术，与 🤗 Transformers 和 🤗 Accelerate 无缝集成。这使得能够使用来自 Transformers 的最流行和高性能的模型，以及 Accelerate 的简单性和可扩展性。以下是目前支持的 PEFT 方法:

LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
Prefix Tuning: [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf)
P-Tuning: [GPT Understands](https://arxiv.org/pdf/2103.10385.pdf), [Too](https://arxiv.org/pdf/2103.10385.pdf)

## 使用 🤗 PEFT 训练您的模型
### 引进必要的库
