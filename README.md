# llm-learning-journey
大模型到底是啥，底层原理是啥？

## 学习指导
目标：**AI工程化 / LLM应用开发**方向，是目前市场上非常热门的领域。

这个岗位**不是**让你去训练底座大模型（Pre-training），而是**如何用好模型、构建平台、解决业务落地问题**。

基于此，制定了如下的进阶学习路线，分为**基础巩固、核心转型、高阶实战、加分项**四个阶段：
### 第一阶段：夯实后端与系统基础（对应 JD 第2点）

JD 明确要求 Java/Python 和 Linux 能力，这是地基。

**1. 语言能力 (Java 或 Python 选其一，建议 Python 补强)**
*   **如果你主要用 Java：**
    *   **重点：** 高并发、JVM 调优、分布式架构（Spring Cloud/Dubbo）。因为“平台开发”通常涉及高吞吐量的API服务。
    *   **补强：** 必须学习 Python 基础。虽然核心业务用 Java，但 AI 生态（PyTorch, LangChain, HF）主要在 Python。你需要能读懂 Python 代码并进行跨语言调用（如 gRPC, Sidecar 模式）。
*   **如果你主要用 Python：**
    *   **重点：** 异步编程（Asyncio）、高性能 Web 框架（FastAPI）、类型系统（Pydantic）。
    *   **补强：** 了解企业级架构设计，设计模式。

**2. Linux 环境开发与调试**
*   **熟练操作：** Shell 脚本编写，常用命令（grep, awk, sed）。
*   **调试能力：** 能够分析 CPU/内存 占用（top, htop, pidstat），网络抓包（tcpdump, wireshark），日志分析。
*   **容器化：** Docker 的深度使用，Kubernetes (K8s) 的基本操作和部署（因为大模型应用通常部署在容器集群中）。

---

### 第二阶段：LLM 应用开发核心（对应 JD 第3点前半部分）

这是转型的关键，你需要从传统 CRUD 工程师转变为 AI 应用工程师。

**1. 理论基础**
*   理解 **Transformer** 架构（Encoder/Decoder 区别）。
*   理解 **Token**（分词）、**Embedding**（向量化）、**Temperature**（温度参数）等基本概念。
*   **Prompt Engineering（提示词工程）：** 掌握 CoT (Chain of Thought), Few-shot prompting。

**2. RAG（检索增强生成）技术栈 —— *JD 核心考点***
*   **原理：** 也就是“外挂知识库”。学习如何将私有数据喂给大模型。
*   **向量数据库：** 学习使用 Milvus, ChromaDB, Pinecone, 或 Elasticsearch 的向量检索功能。
*   **数据处理 (ETL)：** 学习 Unstructured 等库，掌握 PDF/Word/Markdown 的解析与 **Chunking（切片）** 策略（按字符切、按语义切）。
*   **框架应用：** 深度掌握 **LangChain** 或 **LlamaIndex**。这是目前最主流的 RAG 编排框架。

---

### 第三阶段：Agent 与高阶工程化（对应 JD 第3点后半部分）

JD 提到了 Tool Calling 和 Agent 框架，这是目前 AI 应用的前沿。

**1. Tool Calling (Function Calling)**
*   学习 OpenAI 格式的 Function Calling 协议。
*   **实战：** 写一个 Demo，让大模型能够“调用”你写的 Python 函数（例如：查询天气、查询数据库、调用外部 API），并根据结果生成回答。

**2. Agent（智能体）开发**
*   **设计模式：** 理解 **ReAct** (Reasoning + Acting) 模式，理解 Plan-and-Solve。
*   **框架：**
    *   **LangGraph:** 目前 LangChain 推崇的构建复杂 Agent 的图框架（非常重要）。
    *   **AutoGPT / CrewAI:** 了解多 Agent 协作的概念。
*   **实战目标：** 构建一个能够自主规划任务的 Agent（例如：给定一个模糊目标“帮我写一份关于百炼的市场分析”，它能自动搜索、阅读网页、总结并生成报告）。

**3. 模型部署与推理优化**
*   **工具：** 学习 **vLLM**, **TGI (Text Generation Inference)**, 或 **Ollama**。
*   **量化：** 了解 FP16, INT8, GPTQ, AWQ 等量化技术，如何在显存受限的情况下跑通模型。
*   **微调 (Optional but Good):** 了解 **LoRA / QLoRA** 微调流程，如何基于开源模型（如 Qwen, Llama 3）微调一个特定垂直领域的模型。

---

### 第四阶段：开源与生态建设（对应 JD 第4点）

这是进入大厂（尤其是阿里云这种做平台的公司）的“敲门砖”。

**1. 拥抱开源**
*   **阅读源码：** 不要只调包，去读 LangChain, AutoGPT 或阿里开源的 **Qwen (通义千问)** 仓库的源码。
*   **贡献代码：** 给这些项目提 PR（Pull Request），哪怕是修复文档错误、增加一个小工具的集成，或者修复一个 Bug。JD 明确说了“特别欢迎”有贡献经验者。

**2. 熟悉阿里/百炼生态**
*   去阿里云官网注册试用 **“百炼 (Model Studio)”** 平台。
*   熟悉 **Qwen (通义千问)** 系列模型的能力。
*   如果在面试中能说出：“我用过百炼平台，觉得某个功能设计得很好，但某个 API 如果这样改进会更方便开发”，你会非常加分。

---

### 总结：推荐的学习与项目清单

为了在 3-6 个月内达到要求，建议完成以下 2 个实战项目：

**项目一：企业级 RAG 知识库助手**
*   **技术栈：** Python/Java + LangChain + Milvus + Qwen/OpenAI API + Streamlit/React。
*   **功能：** 上传公司 PDF 手册，能够通过对话精准回答问题，并给出引用来源。
*   **难点攻克：** 解决切片不准的问题，优化检索召回率（Re-ranking）。

**项目二：具备 Tool Calling 能力的运维 Agent**
*   **技术栈：** LangGraph + FastAPI + Linux Shell。
*   **功能：** 对话式运维。用户输入“帮我查一下生产环境 CPU 最高的进程”，Agent 自动解析意图，生成 Shell 命令，在沙箱执行，并返回分析结果。
*   **亮点：** 体现了 JD 中的“Linux下开发调试” + “Tool Calling” + “Agent框架”。

**核心心态：**
JD 里提到了“热爱技术，持续动手试新东西”。AI 领域通过 paper 到 code 的速度极快，保持对 HuggingFace Trending 和 Twitter/X 技术圈的关注，展示你对新技术的敏感度至关重要。