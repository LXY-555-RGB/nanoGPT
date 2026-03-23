# nanoGpt

## 一、简介

nanoGPT 是一款主打简洁、高效的中等规模 GPT 模型训练 / 微调框架，由 minGPT 重构而来，核心设计理念是优先保证实用性而非仅用于教学。

**核心特点**

​	极简代码架构：核心逻辑高度精简，train.py 仅约 300 行代码实现训练循环，model.py 约 300 行完成 GPT 模型定义，且支持加载 OpenAI 发布的 GPT-2 预训练权重，易于理解和二次开发。

​	高效训练能力：可复现 GPT-2（124M 参数）在 OpenWebText 数据集上的训练效果，单节点 8 张 A100 40GB GPU 仅需约 4 天即可完成训练，同时支持单机单卡、单机多卡、多机多卡等多种训练部署方式。

​	灵活适配性：兼容不同硬件环境，既支持高性能 GPU 集群，也适配 MacBook（包括 Apple Silicon 芯片）、普通 CPU 等低成本设备，可通过调整超参数（如模型层数、上下文长度、批次大小等）适配不同算力。

​	完整的流程支持：覆盖数据预处理（如 Shakespeare 字符级数据集、OpenWebText 数据集的下载与 token 化）、模型训练、微调、采样推理全流程，还提供基准测试（bench.py）、损失评估等辅助功能。

**核心功能**

​	基础训练：支持从零训练字符级 GPT 模型（如基于莎士比亚文本），也可复现 GPT-2 全量训练流程，输出的模型 checkpoint 可直接用于采样生成文本。

​	模型微调：支持基于 GPT-2 预训练模型进行下游数据集微调，仅需少量算力即可完成（如单 GPU 几分钟内完成莎士比亚文本微调），有效降低领域适配成本。

​	灵活采样：提供 sample.py 脚本，可从训练 / 微调后的模型或 OpenAI 官方 GPT-2 模型（如 gpt2-xl）中采样生成文本，支持自定义起始提示、生成长度等参数。

​	性能优化：默认集成 PyTorch 2.0 torch.compile() 特性，可显著降低训练迭代耗时，同时支持分布式数据并行（DDP）、学习率衰减、Dropout 正则化等优化策略。

**适用场景**

适合深度学习开发者快速上手 GPT 模型训练、验证模型改进思路，或针对小数据集进行定制化微调，尤其适合希望深入理解 GPT 底层实现、轻量化部署 GPT 类模型的场景。

## 二、 训练代码分析

​	该脚本实现了 GPT 模型从 “从零训练 / 断点续训 / 加载 GPT-2 预训练权重” 到 “训练过程监控、学习率调度、模型保存” 的全流程，默认适配 OpenWebText 数据集（GPT-2 的训练数据集），可通过配置灵活调整模型规模、训练策略、硬件适配等。

| 类别       | 核心参数                                                  | 作用说明                                                     |
| ---------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| I/O 与日志 | out_dir/eval_interval/wandb_log                           | 模型保存路径、评估间隔、是否用 WandB 可视化训练日志          |
| 数据配置   | dataset/batch_size/block_size/gradient_accumulation_steps | 数据集名称、微批次大小、上下文窗口长度、梯度累积步数（模拟大批次） |
| 模型架构   | n_layer/n_head/n_embd/dropout/bias                        | 模型层数、注意力头数、嵌入维度、dropout 率、是否在层归一化 / 线性层用偏置 |
| 优化器     | learning_rate/weight_decay/beta1/beta2/grad_clip          | AdamW 优化器参数、梯度裁剪阈值                               |
| 学习率调度 | warmup_iters/lr_decay_iters/min_lr                        | 预热步数、衰减总步数、最小学习率（余弦衰减策略）             |
| 分布式训练 | backend/ddp相关环境变量                                   | DDP 通信后端（默认 nccl）、多卡训练的进程配置                |
| 系统适配   | device/dtype/compile                                      | 训练设备（cuda/cpu/mps）、数据类型（bf16/fp16/fp32）、是否编译模型加速 |



#### 1.分布式训练初始化（DDP）

检测环境变量RANK判断是否为 DDP 训练，若为 DDP 则初始化进程组、分配 GPU 设备、设置主进程（仅主进程做日志 / 保存模型）；

自动缩放梯度累积步数（按进程数均分），保证多卡训练的总批次大小与单卡一致。

#### **2.数据加载（极简版 DataLoader）**

基于np.memmap加载二进制数据集（train.bin/val.bin），避免全量加载占用内存；

随机采样batch_size个起始位置，生成输入x（上下文）和标签y（下一个 token），支持 CUDA 异步数据传输（pin_memory）。

**3.模型初始化**

支持三种初始化方式：

scratch：从零初始化模型，自动读取数据集的meta.pkl获取词汇表大小，默认用 GPT-2 的 50304；

resume：从out_dir的ckpt.pt断点续训，强制对齐模型核心架构参数（层数 / 头数等）；

gpt2*：加载 OpenAI 官方 GPT-2 预训练权重（支持 gpt2/gpt2-medium 等）。

可选裁剪模型的block_size（上下文长度），适配小数据集。

#### **4.训练核心逻辑**

损失评估（estimate_loss）

无梯度计算（torch.no_grad()），在训练 / 验证集上跑eval_iters个批次，取平均损失作为当前模型性能；

评估时模型切到eval模式，避免 dropout 影响。

#### **学习率调度（get_lr）**

预热阶段（warmup_iters）：线性提升学习率；

衰减阶段：余弦衰减至min_lr；

超出衰减步数后固定为min_lr。

#### **训练循环**

梯度累积：将gradient_accumulation_steps个微批次的梯度累加，模拟大批次训练；

混合精度训练：基于GradScaler实现 fp16/bf16 混合精度，加速训练且节省显存；

DDP 梯度同步：仅在最后一个微批次同步梯度，减少通信开销；

日志与保存：主进程按eval_interval输出训练 / 验证损失，保存最优模型（按验证损失）或每次评估都保存；

性能监控：计算模型浮点运算利用率（MFU），反映硬件利用效率。

#### **模型编译与优化**

支持 PyTorch 2.0 的torch.compile编译模型，提升训练速度；

梯度裁剪（grad_clip）防止梯度爆炸；

训练后释放 DDP 进程组（若启用）。



<img src="assets\mfu.png">

灵活性：支持从零训练、断点续训、加载 GPT-2 预训练权重，超参数可通过命令行 / 配置文件覆盖；

高效性：支持混合精度、梯度累积、模型编译、DDP 分布式训练，最大化硬件利用率；

鲁棒性：包含损失评估、学习率调度、梯度裁剪、模型保存等全流程训练保障；

易用性：适配常见的 GPU 训练环境，支持 WandB 可视化，日志清晰。

### 2.1 莎士比亚

导入依赖：文件操作（os）、序列化（pickle）、下载数据（requests）、数值计算（numpy）。

```python
# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")
```

下载数据集：如果本地没有 input.txt，则从指定 URL 下载莎士比亚的小数据集并保存。

读取数据：将文本全部读入内存，并打印数据集的字符总数（示例中约 111.5 万字符）。

```python
# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")
```

构建字符表：提取文本中所有唯一字符并排序，得到词汇表（示例中共 65 个唯一字符，包含空格、标点、大小写字母等）。

打印词汇表和词汇量，方便验证数据范围。

```python
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }  # 字符→整数的映射（encode）
itos = { i:ch for i,ch in enumerate(chars) }  # 整数→字符的映射（decode）
def encode(s):
    return [stoi[c] for c in s]  # 编码函数：字符串→整数列表
def decode(l):
    return ''.join([itos[i] for i in l])  # 解码函数：整数列表→字符串
```

构建编码 / 解码映射：

​	stoi（string to integer）：每个唯一字符对应一个唯一整数 ID；

​	itos（integer to string）：整数 ID 反向映射回字符；

定义编码 / 解码函数：实现「字符串↔整数列表」的双向转换，是字符级模型的核心映射逻辑。

```python
# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]  # 前90%作为训练集
val_data = data[int(n*0.9):]    # 后10%作为验证集

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
```

划分训练 / 验证集：按 9:1 的比例拆分文本数据；

编码数据：将训练集和验证集的字符串转换为整数列表，并打印各自的 token 数（字符级模型中，1 个字符 = 1 个 token）。

```python
# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
```

保存为二进制文件：

将整数列表转换为uint16类型的 numpy 数组（节省空间，65 个字符仅需 7 位，uint16 足够）；

写入train.bin和val.bin，二进制格式比文本格式更高效，适合模型读取。

```python
# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
```

train.bin/val.bin：二进制文件，存储编码后的整数 ID 序列，可通过np.fromfile快速加载；

meta.pkl：序列化文件，存储词汇表大小、字符↔整数的映射关系，用于后续编码 / 解码；

input.txt：原始的莎士比亚文本数据集（下载后生成）。

<img src="assets\mu2.png">

​	在完成数据集训练后可进行生成，其下为结果：

<img src="assets\ssby.png">





### 2.2 诗词

按上述步骤对诗词文本进行训练，以下为结果：

<img src="assets\sc.png">

### 2.3天龙八部

对天龙八部txt进行训练，以下为结果：

<img src="assets\tianlong.png">

## 三、总结

​		训练了诗词生成 GPT和天龙八部风格 GPT，二者遵循完全一致的GPT 定制训练三步骤，是 nanoGPT 的核心落地逻辑：
数据预处理：将原始文本（诗词 / 小说）按 9:1 切分训练 / 测试集→用 GPT-2 的 BPE 分词器（tiktoken）将文本编码为 token 整数→保存为二进制 bin 文件（提升训练读取效率）；
模型训练：通过配置文件定义 GPT 超参数（层数、注意力头数、上下文长度等）→基于 PyTorch 的 GPT 核心架构（Transformer Decoder-only）初始化模型→在自定义数据集上做自回归语言模型训练（以 “预测下一个 token” 为目标）；
自回归推理：将输入提示词编码为 token→模型预测下一个 token 的概率分布→选概率最大的 token 拼接至输入→循环至达到最大长度→解码为文本，完成生成。

**技术重点**
		模型架构：纯Transformer Decoder-only架构（GPT 的核心架构），极简实现（model.py 仅约 300 行），无复杂封装，可直接查看层归一化、多头注意力、前馈网络等核心组件的代码实现；
分词技术：使用 OpenAI 的GPT-2 BPE 分词（tiktoken），是大语言模型的核心基础，理解 “文本→token→整数” 的编码逻辑；
训练范式：自回归语言建模（CLM），大模型最基础的预训练 / 微调范式，目标是最小化 “预测下一个 token” 的交叉熵损失；
超参数调优：针对不同硬件（GPU/CPU）和数据集（短文本 / 长文本）适配超参数（batch_size、block_size、n_layer 等），理解 “硬件资源 - 模型大小 - 数据集特性” 的匹配逻辑。

**适用场景**
		学习场景：大模型初学者理解 GPT 底层原理的最佳实操工具，代码简洁、无冗余，可直接修改 / 调试核心模块；
定制小模型：针对垂直小数据集（如诗词、小说、行业话术）训练轻量定制 GPT，部署在本地 CPU / 低配 GPU，满足特定场景的文本生成需求；
技术验证：快速验证新的数据集、超参数、模型结构对 GPT 生成效果的影响，做快速原型验证。

**案例亮点与局限性**
		亮点：极简、可解释、易复现，剥离了工业级大模型的复杂封装（如分布式训练、量化、优化器封装），聚焦核心原理；硬件要求低，CPU 即可完成训练，入门门槛极低。
		局限性：模型效果受限于数据集大小和模型参数量（本案例为百万级参数量），生成文本易出现逻辑混乱、重复；仅支持字符 / 小 token 级生成，无工业级大模型的复杂采样策略（如 Top-K、Top-P、温度调节）。

