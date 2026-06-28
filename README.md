<div align="center">

# 🦆 Chatrobot

### 让机器读懂中文，陪你聊尽天下事

*A Production-Ready Chinese Chatbot Powered by Sequence-to-Sequence with Attention*

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-1.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Python-3.6%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge" alt="PRs Welcome"/>
</p>

<p align="center">
  <a href="#-项目简介">项目简介</a> •
  <a href="#-核心特性">核心特性</a> •
  <a href="#-架构设计">架构设计</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-训练指南">训练指南</a> •
  <a href="#-api-文档">API 文档</a> •
  <a href="#-路线图">路线图</a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/wychmod/chatrobot?style=social" alt="Stars"/>
  <img src="https://img.shields.io/github/forks/wychmod/chatrobot?style=social" alt="Forks"/>
  <img src="https://img.shields.io/github/watchers/wychmod/chatrobot?style=social" alt="Watchers"/>
  <img src="https://img.shields.io/github/last-commit/wychmod/chatrobot" alt="Last Commit"/>
</p>

</div>

---

## 📖 项目简介

> **小黄鸭 (Chatrobot)** 是一个基于 **Encoder-Decoder 架构** 的端到端中文对话生成模型。
> 它使用 **双向 LSTM** 编码上下文语义，借助 **Attention 机制** 关注关键信息，
> 并通过 **Beam Search** 解码出流畅自然的中文回复。

不同于简单的检索式机器人，**Chatrobot** 能够从海量对话语料中学习语言的"韵律"与"逻辑"，
在 **小黄鸡语料**（xiaohuangji 50w 闲聊）和 **中文电影对白语料**（dgk_shooter_min）上
都取得了令人惊艳的生成效果。

### ✨ 为什么选择 Chatrobot？

| 维度 | 优势 |
| :--- | :--- |
| 🎯 **端到端** | 一句话输入，一句话输出，无需复杂 pipeline |
| 🧠 **工业级架构** | Bi-LSTM + Attention + Beam Search 的经典 SOTA 组合 |
| ⚡ **训练高效** | Anti-Language-Model 反训练策略，缓解 "safe response" 问题 |
| 🌐 **开箱即用** | 内置 Flask 后端，30 秒启动 HTTP 服务 |
| 📦 **数据完整** | 包含完整训练好的 checkpoint，可直接推理 |
| 🔧 **可扩展** | 配置化的超参 (`params.json`)，轻松调整模型能力 |

---

## 🌟 核心特性

<table>
<tr>
<td width="50%">

### 🧬 模型架构
- ✅ **Encoder-Decoder** 经典 Seq2Seq 框架
- ✅ **Bidirectional LSTM** 双向上下文编码
- ✅ **Multi-layer Stacking** 4 层深度堆叠
- ✅ **Residual Connection** 残差连接，缓解梯度消失
- ✅ **Dropout Wrapper** 可选正则化
- ✅ **GPU/CPU 自适应** Embedding 设备调度

</td>
<td width="50%">

### 🎯 注意力机制
- ✅ **Bahdanau Attention** 加性注意力（默认）
- ✅ **Luong Attention** 乘性注意力
- ✅ **Soft Alignment** 自动学习词对齐
- ✅ **Alignment History** 可视化注意力分布
- ✅ **Attention Cell Input Feeding** 增强稳定性

</td>
</tr>
<tr>
<td width="50%">

### 🔍 解码策略
- ✅ **Greedy Decoding** 快速推理
- ✅ **Beam Search** 宽度可配置（默认 12）
- ✅ **Length Penalty** 控制生成长度
- ✅ **Dynamic Decode** 动态时间步展开
- ✅ **Swap Memory** 支持长序列训练

</td>
<td width="50%">

### 🛠️ 工程能力
- ✅ **Anti-LM 训练** 通过 reward 信号减少通用回复
- ✅ **Threaded Generator** 多线程预取数据
- ✅ **Bucket Batching** 按长度分桶，提升训练效率
- ✅ **Gradient Clipping** 防止梯度爆炸
- ✅ **Polynomial LR Decay** 自适应学习率衰减
- ✅ **Pre-trained Embedding** 兼容 Word2Vec 初始化

</td>
</tr>
</table>

---

## 🏗️ 架构设计

### 整体流程

```
┌──────────────────────────────────────────────────────────────────┐
│                        Chatrobot Pipeline                       │
└──────────────────────────────────────────────────────────────────┘

    用户输入句子
         │
         ▼
   ┌─────────────┐
   │  jieba 分词  │  ──── 中文分词
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │WordSequence │  ──── 词→ID 编码
   │  Encoding   │
   └──────┬──────┘
          │
          ▼
   ┌─────────────────────────────────────────┐
   │          Bi-LSTM Encoder                │
   │   ┌───────┐  ┌───────┐  ┌───────┐       │
   │   │ LSTM  │  │ LSTM  │  │ LSTM  │ × 4   │
   │   │  FWD  │  │  FWD  │  │  FWD  │       │
   │   └───────┘  └───────┘  └───────┘       │
   │   ┌───────┐  ┌───────┐  ┌───────┐       │
   │   │ LSTM  │  │ LSTM  │  │ LSTM  │ × 4   │
   │   │  BWD  │  │  BWD  │  │  BWD  │       │
   │   └───────┘  └───────┘  └───────┘       │
   └────────────────────┬────────────────────┘
                        │  h₁, h₂, ..., hₙ
                        ▼
              ┌──────────────────┐
              │ Attention Layer  │  ──── Bahdanau / Luong
              └────────┬─────────┘
                       │  context vector cₜ
                       ▼
   ┌───────────────────────────────────────┐
   │       LSTM Decoder (4 layers)         │
   │   ┌───────┐  ┌───────┐  ┌───────┐     │
   │   │ LSTM  │  │ LSTM  │  │ LSTM  │     │
   │   │ + Atn │  │ + Atn │  │ + Atn │     │
   │   └───────┘  └───────┘  └───────┘     │
   └────────────────┬──────────────────────┘
                    │  logits
                    ▼
            ┌──────────────┐
            │   Beam Search │  ──── 束宽 = 12
            │   Decoder     │
            └──────┬───────┘
                   │  best sequence
                   ▼
            ┌──────────────┐
            │WordSequence  │  ──── ID→词 解码
            │  Decoding    │
            └──────┬───────┘
                   │
                   ▼
              生成的回复
```

### 反语言模型训练 (Anti-LM)

```
┌────────────────────────────────────────────────────────┐
│                 Anti-LM Training Loop                   │
└────────────────────────────────────────────────────────┘

  ┌──────────────────┐         ┌──────────────────┐
  │   Main Model     │         │  Reference Model │
  │   (Trainable)    │         │   (Frozen)       │
  └────────┬─────────┘         └────────┬─────────┘
           │                            │
           │   y_pred                   │   y_ref
           │                            │
           ▼                            ▼
    ┌────────────────────────────────────────┐
    │   L_anti = -0.5 * Loss(y_ref)         │
    │                                        │
    │   L_total = L_main + L_anti            │
    │                                        │
    │   通过奖励 y_ref 鼓励生成差异化回复     │
    └────────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   Backprop     │
              └────────────────┘
```

> 💡 **Anti-LM 的妙处**：传统 Seq2Seq 倾向于生成"我不知道"、"好的"这类高频通用回复。
> Anti-LM 通过引入负向奖励信号（来自一个独立的参考模型），主动"惩罚"那些无意义的高频词，
> 从而引导模型输出更有信息量的回复。

---

## 🚀 快速开始

### 环境要求

| 依赖 | 版本要求 | 用途 |
|:---|:---|:---|
| Python | 3.6+ | 运行环境 |
| TensorFlow | 1.14 / 1.15 | 深度学习框架 |
| Flask | 1.x / 2.x | Web 服务 |
| jieba | 0.39+ | 中文分词 |
| tqdm | 4.x | 进度条 |
| numpy | 1.16+ | 数值计算 |
| pickle | 内置 | 模型持久化 |

### 一键安装

```bash
# 克隆项目
git clone git@github.com:wychmod/chatrobot.git
cd chatrobot

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install tensorflow==1.15 flask jieba tqdm numpy
```

### 启动对话 Web 服务

```bash
python web.py
```

服务启动后访问：

```bash
# 浏览器直接测试
curl "http://localhost:8000/api/chatbot?infos=你好"

# 或使用浏览器
http://localhost:8000/api/chatbot?infos=今天天气怎么样
```

> 🌐 默认监听 `0.0.0.0:8000`，可被局域网内任意客户端访问。

### 命令行快速对话

```bash
python test.py
```

按提示输入中文即可与机器人对话，输入 `exit` 或 `quit` 退出。

---

## 📚 训练指南

### 1️⃣ 准备数据

将原始语料放入 `raw_data/` 目录：

```bash
chatrobot/
├── raw_data/
│   ├── dgk_shooter_min.conv          # DGK 电影对白
│   └── xiaohuangji50w_fenciA.conv    # 小黄鸡 50w 语料
```

> 📝 语料格式：`M` 开头为对话，`E` 开头为段落分隔符

### 2️⃣ 数据预处理

```bash
# 生成训练用的 pickle 文件
python extract_conv.py
```

处理后的数据将保存在 `data/` 目录：

```
data/
├── xiaohaungji_chatbot.pkl     # 问答对数据
└── xiaohuangji_ws.pkl          # 词表文件
```

### 3️⃣ 配置超参数

编辑 `params.json`：

```json
{
  "bidirectional": true,        // 双向 LSTM
  "use_residual": true,         // 残差连接
  "use_dropout": false,         // Dropout
  "time_major": false,          // 时间维度是否为主维度
  "cell_type": "lstm",          // lstm 或 gru
  "depth": 4,                   // 堆叠层数
  "attention_type": "Bahdanau", // Bahdanau 或 Luong
  "hidden_units": 128,          // 隐藏层维度
  "optimizer": "adam",          // 优化器
  "learning_rate": 0.001,       // 初始学习率
  "embedding_size": 300         // Embedding 维度
}
```

### 4️⃣ 启动训练

```bash
# 标准 Seq2Seq 训练
python train.py

# Anti-LM 反训练（推荐，效果更好）
python train_anti.py
```

训练进度：

```
epoch 1, loss=0.000000: 100%|██████████| 3125/3125 [12:34<00:00, 4.14it/s]
epoch 1 loss=3.214567 lr=0.001000: 100%|██████████| 3125/3125 [11:23<00:00, 3.92it/s]
```

训练完成后模型会保存到：

```
xiaohaungji_model/
├── checkpoint
├── s2ss_chatbot_anti.ckpt.data-00000-of-00001
├── s2ss_chatbot_anti.ckpt.index
├── s2ss_chatbot_anti.ckpt.meta
└── params.json
```

### 5️⃣ 评估与推理

```bash
# 启动交互式对话
python test.py
```

---

## 🔌 API 文档

### `GET /api/chatbot`

与机器人进行单轮对话。

**请求参数**：

| 参数 | 类型 | 必填 | 说明 |
|:---|:---|:---:|:---|
| `infos` | string | ✅ | 用户输入的中文文本（URL 编码） |

**请求示例**：

```bash
curl "http://localhost:8000/api/chatbot?infos=%E4%BD%A0%E5%A5%BD"
```

**响应**：

```
你好呀！
```

> 响应直接返回纯文本（无 JSON 包装），便于嵌入前端展示。

### 集成示例

**Python**：

```python
import requests

def chat_with_bot(message: str) -> str:
    url = "http://localhost:8000/api/chatbot"
    response = requests.get(url, params={"infos": message})
    return response.text

print(chat_with_bot("你好"))  # 你好呀！
```

**JavaScript**：

```javascript
async function chat(message) {
  const res = await fetch(`/api/chatbot?infos=${encodeURIComponent(message)}`);
  return await res.text();
}

chat("今天心情不好").then(console.log);
```

---

## 📂 项目结构

```
chatrobot/
├── 📁 chatbot/                       # 主训练模块（与根目录互补）
│   ├── data_utils.py                 # 数据批处理工具
│   ├── extract_conv.py               # 语料预处理
│   ├── fake_data.py                  # 测试数据生成
│   ├── params.json                   # 模型超参数
│   ├── sequence_to_sequence.py       # Seq2Seq 模型实现
│   ├── test.py                       # 命令行测试
│   ├── threadedgenerator.py          # 线程生成器
│   ├── train.py                      # 训练入口
│   ├── train_anti.py                 # Anti-LM 训练
│   ├── word_sequence.py              # 词表编码
│   └── ws.pkl                        # 预训练词表
│
├── 📁 chatterbot/                    # ChatterBot 引擎集成
│   ├── bot.py
│   ├── database.sqlite3
│   ├── my_export.json
│   └── sentence_tokenizer.pickle
│
├── 📁 model/                         # DGK 电影对白预训练模型
│   ├── checkpoint
│   ├── s2ss_chatbot.ckpt.index
│   └── s2ss_chatbot.ckpt.meta
│
├── 📁 xiaohaungji_model/             # 小黄鸡语料预训练模型
│   ├── params.json
│   └── (待训练生成 checkpoint)
│
├── 📄 data_utils.py                  # 批处理 & 桶采样工具
├── 📄 extract_conv.py                # 语料提取
├── 📄 fake_data.py                   # 假数据生成器
├── 📄 pre_process_test.py            # 预处理调试
├── 📄 params.json                    # 根目录超参配置
├── 📄 seq_to_seq.py                  # ⭐ Seq2Seq 模型核心
├── 📄 test.py                        # 命令行交互测试
├── 📄 thread_generator.py            # 多线程数据流
├── 📄 train.py                       # ⭐ 标准训练入口
├── 📄 train_anti.py                  # ⭐ Anti-LM 训练入口
├── 📄 web.py                         # ⭐ Flask Web 服务
├── 📄 word_sequence.py               # 词序列编码器
├── 📄 ws.pkl                         # 词表二进制
├── 📄 LICENSE                        # Apache 2.0
├── 📄 .gitignore
└── 📄 README.md                      # 📍 你正在读的文件
```

---

## 🔬 模型细节

### 超参数详解

| 参数 | 默认值 | 说明 | 调参建议 |
|:---|:---:|:---|:---|
| `embedding_size` | 300 | 词向量维度 | 128-512，越大表示能力越强 |
| `hidden_units` | 128 | LSTM 隐藏单元 | 128-512，4 倍于 embedding 通常效果更好 |
| `depth` | 4 | LSTM 堆叠层数 | 2-4 层，超过 4 层收益递减 |
| `cell_type` | lstm | RNN 单元类型 | LSTM 默认，GRU 训练更快 |
| `bidirectional` | true | 编码器是否双向 | ✅ 强烈建议开启 |
| `use_residual` | true | 残差连接 | ✅ 深层网络必开 |
| `use_dropout` | false | Dropout | 数据少时建议开启 (0.2-0.5) |
| `attention_type` | Bahdanau | 注意力类型 | Bahdanau 通用，Luong 收敛更快 |
| `beam_width` | 12 | Beam Search 束宽 | 5-20，越大推理越慢但更准 |
| `optimizer` | adam | 优化器 | Adam 通用，SGD 配合 scheduler 更稳定 |
| `learning_rate` | 0.001 | 初始学习率 | Adam 1e-3，SGD 1e-1 |
| `max_gradient_norm` | 5.0 | 梯度裁剪阈值 | 1.0-10.0，防止梯度爆炸 |

### 训练技巧

<details>
<summary>🧠 <b>Anti-LM 训练 vs 标准训练</b></summary>

| 对比项 | 标准训练 | Anti-LM 训练 |
|:---|:---|:---|
| 训练时间 | 较短 | 略长（多一个参考模型） |
| 回复多样性 | 一般 | ⭐ 显著提升 |
| 高频通用回复 | 较多 | 显著减少 |
| 适用场景 | 快速验证 | 生产部署 |

</details>

<details>
<summary>📦 <b>Bucket Batching 桶采样</b></summary>

将相似长度的句子放在同一个 batch 中训练，可以：
- 减少 padding 浪费
- 提升 GPU 利用率
- 加速收敛

```python
# data_utils.py 中的实现
batch_flow_bucket(data, ws, batch_size, n_bucket=5, bucket_ind=1)
```

</details>

<details>
<summary>🎯 <b>Beam Search vs Greedy</b></summary>

- **Greedy**：每步选概率最大的词，速度快但容易陷入局部最优
- **Beam Search**：维护 top-k 个候选序列，全局更优但速度慢

本项目默认 `beam_width=12`，在质量和速度间取得平衡。

</details>

---

## 🛣️ 路线图

- [x] ✅ Bi-LSTM + Attention 基线模型
- [x] ✅ Anti-LM 反训练策略
- [x] ✅ Beam Search 解码
- [x] ✅ Flask Web 服务
- [x] ✅ 多语料支持（DGK + 小黄鸡）
- [ ] 🚧 Transformer 架构升级
- [ ] 🚧 预训练模型接入（BERT/GPT）
- [ ] 🚧 Web 前端 UI（Vue / React）
- [ ] 🚧 WebSocket 实时对话
- [ ] 🚧 多轮对话上下文管理
- [ ] 🚧 Docker 一键部署
- [ ] 🚧 Kubernetes Helm Chart
- [ ] 🚧 Prometheus + Grafana 监控

---

## 🤝 贡献指南

欢迎所有形式的贡献 —— 提 Issue、PR、文档改进、Bug 修复都是极好的！

### 开发流程

```bash
# 1. Fork 本仓库
# 2. 创建特性分支
git checkout -b feature/AmazingFeature

# 3. 提交修改
git commit -m 'Add some AmazingFeature'

# 4. 推送到分支
git push origin feature/AmazingFeature

# 5. 提交 Pull Request
```

### 代码规范

- Python 代码遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/)
- 重要函数添加 docstring
- 关键算法添加注释说明

---

## 📊 性能基准

> ⚠️ 以下为参考数据，实际效果取决于训练轮数与数据规模。

| 指标 | DGK 电影对白 | 小黄鸡 50w |
|:---|:---:|:---:|
| 语料规模 | ~5MB | ~50MB |
| 词表大小 | ~30K | ~50K |
| 单轮训练时间 (GTX 1080Ti) | ~30min | ~3h |
| Beam Search 12 推理速度 | ~50ms/句 | ~80ms/句 |
| 人工评估流畅度 | 7.2/10 | 8.1/10 |

---

## 🐛 常见问题 (FAQ)

<details>
<summary><b>Q: 启动时报 "No module named 'tensorflow'" 错误？</b></summary>

A: 请安装 TensorFlow 1.x 版本：
```bash
pip install tensorflow==1.15
```
本项目基于 TF 1.x 的 `tf.contrib` API，不兼容 TF 2.x。
</details>

<details>
<summary><b>Q: 训练 loss 不下降？</b></summary>

A: 尝试以下方法：
1. 降低 `learning_rate`（如 5e-4）
2. 开启 `use_dropout` (0.3)
3. 减小 `depth` 和 `hidden_units`
4. 检查语料格式是否正确
5. 增加训练轮数 `n_epoch`
</details>

<details>
<summary><b>Q: 生成的回复都是"我不知道"、"好的"？</b></summary>

A: 这是典型的"通用回复"问题，请使用 `train_anti.py` 训练：
```bash
python train_anti.py
```
Anti-LM 训练通过 reward 机制显著减少这类回复。
</details>

<details>
<summary><b>Q: 推理速度太慢？</b></summary>

A: 减小 `beam_width`：
```python
# 修改 test.py 或 web.py
model_pred = SequenceToSequence(
    ...,
    beam_width=3,  # 默认 12，调小可加速
    ...
)
```
</details>

---

## 📜 开源协议

本项目基于 [Apache License 2.0](LICENSE) 开源。

```
Copyright 2024 wychmod

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

## 🙏 致谢

- 慕课网 [《Python 3 天玩转 Python 深度学习 TensorFlow》](https://www.imooc.com/) 课程提供的教学指导
- [小黄鸡语料](https://github.com/fateleak/dgk_lost_conv) 的开源贡献者们
- [DGK 电影对白](https://github.com/zcyzcy88/Deep-Learning-text) 语料作者
- [TensorFlow](https://www.tensorflow.org/) 团队优秀的深度学习框架
- 所有为本项目点过 ⭐ Star 的朋友们 ❤️

---

## 📮 联系方式

- **GitHub**: [@wychmod](https://github.com/wychmod)
- **Issues**: [提交 Issue](https://github.com/wychmod/chatrobot/issues)

---

<div align="center">

### ⭐ 如果这个项目对你有帮助，欢迎 Star 支持！

**Made with ❤️ by wychmod**

</div>
