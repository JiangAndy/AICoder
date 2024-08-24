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

## 训练feature
Accelerate 提供了额外的功能，例如梯度累积 (gradient accumulation)、梯度裁剪 (gradient clipping)、混合精度训练 (mixed precision training)等，您可以将其添加到脚本中以改进训练。

### 梯度累积
梯度累积使您能够在更新权重之前通过累积多个批次的梯度来获取更大的等效 `batch_size`。这对于解决显存对 `batch_size` 的限制很有用。

要在 Accelerate 中启用此功能，请在加速器类中指定 `gradient_accumulation_steps` 参数，并在脚本中添加 `accumulate()` 上下文管理器：

```python
+ accelerator = Accelerator(gradient_accumulation_steps=2)
  model, optimizer, training_dataloader = accelerator.prepare(
      model, optimizer, training_dataloader
  )

  for input, label in training_dataloader:
+     with accelerator.accumulate(model):
          predictions = model(input)
          loss = loss_function(predictions, label)
          accelerator.backward(loss)
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()

```

### 梯度裁剪
梯度裁剪是一种防止“梯度爆炸”的技术，Accelerate 提供以下两种方法：

- `clip_grad_value_`：将可迭代参数的梯度裁剪为指定值。 梯度就地修改（in-place）。
  - `parametres`：可迭代的张量或单个张量，其梯度将归一化
  - `clip_value`：梯度的阈值。梯度被限制在范围内
- `clip_grad_norm_`：范数是对所有梯度一起计算的。梯度就地修改。
  - `parameters`：可迭代的张量或单个张量，其梯度将归一化
  - `max_norm`：梯度的最大范数
  - `norm_type`：float，默认为2.0，用的 p-范数的类型。inf表示无穷范数。

```python
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=2)
dataloader, model, optimizer, scheduler = accelerator.prepare(
dataloader, model, optimizer, scheduler
)

for input, target in dataloader:
     optimizer.zero_grad()
     output = model(input)
     loss = loss_func(output, target)
     accelerator.backward(loss)
     if accelerator.sync_gradients:
     # 二者取其一：
    	accelerator.clip_grad_value_(model.parameters(), clip_value)
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        
     optimizer.step()

```

### 混合精度训练
混合精度通过使用 fp16（半精度）等较低精度的数据类型来计算梯度，从而加速训练。要想使用 Accelerate 获得最佳性能，应在模型内部计算损失（如在 Transformers 模型中），因为模型外部的计算是以全精度进行的。

设置要在 `accelerater` 中使用的混合精度类型，然后使用 `autocast()` 上下文管理器将值自动转换为指定的数据类型。

```python
from accelerate import Accelerator
+ accelerator = Accelerator(mixed_precision="fp16")

+ with accelerator.autocast():
      loss = complex_loss_function(outputs, target):

```

## 保存和加载
训练完成后，加速还可以保存和加载模型，或者您还可以保存模型和优化器状态（optimizer state），这对于恢复训练很有用。

### 模型
所有过程完成后，在保存模型前使用 `unwrap_model()` 方法解除模型的封装，因为训练开始前执行的 `prepare()` 方法将模型封装到了适合的分布式训练接口中。如果不解除对模型的封装，保存模型状态字典的同时也会保存大模型中任何潜在的额外层，这样就无法将权重加载回基础模型中。

使用 `save_model()` 方法来解包并保存模型状态字典。此方法还可以将模型保存到切片检查点 `sharded checkpoints` 或`safetensors`格式中。

```python
accelerator.wait_for_everyone()
accelerator.save_model(model, save_directory)
```

对于 `Transformers` 库中的模型，请使用 `save_pretrained` 方法保存模型，以便可以使用 `from_pretrained` 方法重新加载。

```python
from transformers import AutoModel

unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    "path/to/my_model_directory",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)

model = AutoModel.from_pretrained("path/to/my_model_directory")
```

要加载权重，请在加载权重之前先使用 `unwrap_model()` 方法解包模型。所有模型参数都是对张量的引用，因此这会将您的权重加载到模型中。
```python
unwrapped_model = accelerator.unwrap_model(model)
path_to_checkpoint = os.path.join(save_directory,"pytorch_model.bin")
unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))
```

#### 切片检查点
设置 `safe_serialization=True` 将模型保存为 `safetensor` 格式。

```python
accelerator.wait_for_everyone()
accelerator.save_model(model, save_directory, max_shard_size="1GB", safe_serialization=True)
```

要加载分片检查点或 `safetensor` 格式的检查点，请使用 `load_checkpoint_in_model()` 方法。此方法允许您将检查点加载到特定设备上。
```python
load_checkpoint_in_model(unwrapped_model, save_directory, device_map={"":device})
```

### 状态
在训练过程中，你可能希望保存模型、优化器、随机生成器以及学习率调度器的当前状态，以便在同一个脚本中恢复它们。你应该在脚本中添加 `save_state()` 和 `load_state()` 方法来保存和加载状态。

任何其他需要存储的有状态项目都应使用 `register_for_checkpointing()` 方法注册，以便保存和加载。传递给此方法的每个要存储的对象都必须具有 `load_state_dict` 和 `state_dict` 函数。

## 执行进程
在使用分布式训练系统时，管理跨 `GPU` 执行流程的方式和时间非常重要。有些进程比其他进程完成得更快，有些进程在其他进程尚未完成时就不应开始。Accelerate 提供了用于协调进程执行时间的工具，以确保所有设备上的一切保持同步。

### 在一个进程上执行
某些代码只需在特定机器上运行一次，如打印日志语句或只在本地主进程上显示一个进度条。

#### statement
应使用 `accelerator.is_local_main_process` 来指示只应执行一次的代码。

  `accelerator.is_local_main_process` ：
  - 用于判断当前进程是否是本地节点（服务器）上的主进程，
  - 如果你的训练任务在多台服务器上运行，每台服务器都有一个主进程。`is_local_main_process()` 如果返回 `True`，表示当前进程是本地节点上的主进程。
  - 通常，你可以在本地节点的主进程上执行一些只需执行一次的操作，例如初始化数据、加载预训练模型等。

```python
from tqdm.auto import tqdm

progress_bar = tqdm(
    range(args.max_train_steps), 
    disable=not accelerator.is_local_main_process
)
```

还可以使用 `accelerator.is_local_main_process` 包装语句。
```python
if accelerator.is_local_main_process:
    print("Accelerate is the best")
```

还可以指示 Accelerate 在所有进程中都要执行一次的代码，而不管有多少台机器。如果您要将最终模型上传到 Hub，这将非常有用。

  `accelerator.is_main_process`：

  - 这个函数用于判断当前进程是否是整个训练任务中的主进程。
  - 主进程通常负责一些全局操作，例如模型保存、日志记录等。因此，你可以使用 `is_main_process()` 来确保这些操作只在主进程中执行一次。
  - 如果你的训练任务在多台服务器上运行，`is_main_process()` 将返回 `True`，只有一个服务器上的主进程会满足这个条件。

```python
if accelerator.is_main_process:
    repo.push_to_hub()
```


#### function
对于只应执行一次的函数，请使用 `on_local_main_process` 装饰器。
```python
@accelerator.on_local_main_process
def do_my_thing():
    "Something done once per server"
    do_thing_once_per_server()
```

对于只应在所有进程中执行一次的函数，请使用 `on_main_process` 装饰器。
```python
@accelerator.on_main_process
def do_my_thing():
    "Something done once per server"
    do_thing_once()
```

### 在特定进程上执行

Accelerate 还可以执行只应在特定进程或本地进程索引上执行的函数。

使用 `on_process()` 装饰器指定要执行函数的进程索引。

```python
@accelerator.on_process(process_index=0)
def do_my_thing():
    "Something done on process index 0"
    do_thing_on_index_zero()
```

使用 `on_local_process()` 装饰器指定要执行函数的本地进程索引。
```python
@accelerator.on_local_process(local_process_idx=0)
def do_my_thing():
    "Something done on process index 0 on each server"
    do_thing_on_index_zero_on_each_server()
```

### 推迟执行
当同时在多个 `GPU` 上运行脚本时，某些代码的执行速度可能会比其他代码快。在执行下一组指令之前，您可能需要等待所有进程都达到一定程度。例如，在确保每个进程都完成训练之前，您不应该保存模型。

为此，请在代码中添加 `wait_for_everyone()`。这会阻止所有先完成训练的进程继续训练，直到所有剩余进程都达到相同点（如果在单个 `GPU` 或 `CPU` 上运行，则没有影响）。

```python
accelerator.wait_for_everyone()
```

## 启动Accelerate脚本
首先，将训练代码重写为函数，并使其可作为脚本调用。例如：
```python
  from accelerate import Accelerator
  
+ def main():
      accelerator = Accelerator()

      model, optimizer, training_dataloader, scheduler = accelerator.prepare(
          model, optimizer, training_dataloader, scheduler
      )

      for batch in training_dataloader:
          optimizer.zero_grad()
          inputs, targets = batch
          outputs = model(inputs)
          loss = loss_function(outputs, targets)
          accelerator.backward(loss)
          optimizer.step()
          scheduler.step()

+ if __name__ == "__main__":
+     main()

```

使用以下命令快速启动脚本：
```shell
accelerate launch --accelerate-arg {script_name.py} --script-arg1 --script-arg2 ...
```

使用单个 `GPU` ：
```shell
CUDA_VISIBLE_DEVICES="0" accelerate launch {script_name.py} --arg1 --arg2 ...
```

指定要使用的 `GPU` 数量：

```shell
accelerate launch --num_processes=2 {script_name.py} {--arg1} {--arg2} ...
```

使用混合精度在两个 `GPU` 上启动相同的脚本:
```shell
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 {script_name.py} {--arg1} {--arg2} ...
```

要获取可以传入的参数的完整列表，请运行：
```shell
accelerate launch -h
```

从该自定义 `yaml` 文件启动脚本如下所示：

```shell
accelerate launch --config_file {path/to/config/my_config_file.yaml} {script_name.py} {--arg1} {--arg2} ...
```
