# 大模型训练工具之Accelerate

# accelerate加速分布式训练

随着模型变得越来越大，并行性已经成为在有限硬件上训练更大模型和加速训练速度的策略，增加了数个数量级。Hugging Face，提供了🤗 [加速库](https://huggingface.co/docs/accelerate)，以帮助用户在任何类型的分布式设置上轻松训练🤗 Transformers模型，无论是在一台机器上的多个GPU还是在多个机器上的多个GPU。在本教程中，了解如何自定义您的原生PyTorch训练循环，以启用分布式环境中的训练。

## 设置
通过安装🤗 加速开始:

```shell
pip install accelerate
```

然后导入并创建`Accelerator`对象。`Accelerator`将自动检测您的分布式设置类型，并初始化所有必要的训练组件。您不需要显式地将模型放在设备上。

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

## 准备加速
下一步是将所有相关的训练对象传递给`prepare`方法。这包括您的训练和评估DataLoader、一个模型和一个优化器:

```python
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)
```

## 反向传播
最后一步是用🤗 加速的`backward`方法替换训练循环中的典型`loss.backward()`:

```python
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

如您在下面的代码中所见，您只需要添加四行额外的代码到您的训练循环中即可启用分布式训练！

```python
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

## 启动训练
运行以下命令以创建和保存配置文件:
```shell
accelerate config
```

用以下命令启动训练:
```shell
accelerate launch train.py
```
