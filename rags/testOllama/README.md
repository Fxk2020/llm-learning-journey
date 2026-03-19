针对书籍这种长文档（Long-form content），普通的“切块索引”往往效果不佳，因为书籍有上下文依赖。比如书中写“他走进了房间”，如果你只索引这一句话，模型根本不知道“他”是谁，也忘了之前的剧情。

为了达到**“准确检索”**且**“细节完备”**的效果，我为你设计了一套**“父子索引 + 重排序” (Parent-Child / Auto-Merging)** 的专业方案。

### 核心原理：小块检索，大块回答 (Small-to-Big Retrieval)

这是目前处理书籍最高效的策略：
1.  **切分 (Chunking)**：我们将书切成**父块**（比如 512 token）和**子块**（比如 128 token）。
2.  **索引 (Indexing)**：我们只对**子块**进行向量化索引（因为子块语义更集中，更容易被搜到）。
3.  **检索 (Retrieval)**：当搜索匹配到“子块”时，系统会自动把它替换回它所属的**“父块”**。
4.  **生成 (Generation)**：把包含完整上下文的“父块”喂给大模型。

**结果**：检索极度精准（因为匹配的是细节），回答极度全面（因为喂给模型的是上下文）。

---

### 详细实施方案与代码

你需要用到 `llama-index` 的高级功能。请确保你安装了必要的库：
```bash
pip install llama-index-embeddings-huggingface llama-index-llms-openai sentence-transformers
```

#### 完整的 Python 脚本 (`rags/book_indexer.py`)

这段代码集成了**分层切分**、**本地 Embedding**、**向量存储**和**重排序**。

```python
import os
from dotenv import load_dotenv

# 1. 基础设置
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# 加载 API KEY (假设你用 OpenAI 做最后的问答生成，如果用本地 LLM 改这里即可)
load_dotenv()

# --- 配置模型 ---
# Embedding 使用本地免费且强大的 BGE 模型
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# LLM 使用 OpenAI (或者你也可以换成本地 Ollama)
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

# --- 2. 加载与切分书籍 (关键步骤) ---
print("正在加载书籍...")
# 假设你的书在 data 目录下，可以是 PDF, TXT, Markdown
documents = SimpleDirectoryReader("./data").load_data()

# 核心：定义分层切分器
# chunk_sizes: [2048, 512, 128] 意味着它会创建三层结构
# 最小的 128 是叶子节点（用于搜索），中间的 512 和最大的 2048 是父节点（用于提供上下文）
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

print("正在处理节点结构（这可能需要一点时间）...")
nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes) # 我们只索引叶子节点（最小的块）

# --- 3. 构建索引与存储 ---
# 我们需要一个 StorageContext 来保存 父节点和子节点 的映射关系
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes) # 把所有层级的节点都存进去

print("正在构建向量索引...")
index = VectorStoreIndex(
    leaf_nodes, # 向量库里只存叶子节点
    storage_context=storage_context
)

# --- 4. 构建高级检索器 (Auto-Merging) ---
# 这个检索器的作用：搜到叶子节点后，自动“合并”回它的父节点
base_retriever = index.as_retriever(similarity_top_k=10)
retriever = AutoMergingRetriever(
    base_retriever, 
    storage_context, 
    verbose=True # 设置为 True 可以看到它合并的过程
)

# --- 5. 加入重排序 (Reranker) ---
# 使用 BGE Reranker 再次提升准确度
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base", 
    top_n=5 # 最终只给大模型看最准的 5 段上下文
)

# --- 6. 组装查询引擎 ---
query_engine = RetrieverQueryEngine.from_args(
    retriever, 
    node_postprocessors=[reranker]
)

# --- 7. 测试 ---
print("\n=== 系统就绪，开始提问 ===")
question = "这本书里关于主角童年的主要冲突是什么？" 
# 替换成你书里的具体问题

response = query_engine.query(question)
print(f"\n问题: {question}")
print("-" * 30)
print(f"回答: {response}")

# 调试：查看模型到底参考了哪段原文
# print("\n参考原文:")
# for node in response.source_nodes:
#    print(f"--- score: {node.score} ---\n{node.text[:200]}...\n")
```

### 为什么这个方案适合书籍？

1.  **分层结构 (`HierarchicalNodeParser`)**：
    *   书里的逻辑是跨段落的。普通的切分器（比如每 500 字切一刀）经常把一个完整的论述切成两半。
    *   这个方案里，如果检索到了第 10 页的一句话，系统会自动把第 9-11 页的相关内容（父节点）一起抓取出来。这保证了**上下文的完整性**。

2.  **叶子节点索引 (`Leaf Node Indexing`)**：
    *   索引库里存的是极小的片段（128 token）。小片段语义非常单纯，杂音少，更容易被向量检索命中。

3.  **自动合并检索器 (`AutoMergingRetriever`)**：
    *   这是 LlamaIndex 的黑科技。如果它发现检索结果里的几个小片段都属于同一个“父段落”，它就会把这些小片段丢掉，直接把那个更完整的“父段落”拿出来。

4.  **重排序 (`Reranker`)**：
    *   书籍内容往往会有很多重复的词汇（比如主角名字）。Reranker 能分辨出哪一段是真正描述你问题核心的，哪一段只是顺带提到了名字。

### 进阶建议：处理 PDF 的“页眉页脚”干扰

如果你喂的是 PDF 书籍，页眉（书名）和页脚（页码）是最大的噪音，会严重干扰索引。

如果发现检索结果总是包含页码，建议在 `SimpleDirectoryReader` 之前做一步清洗，或者使用 **LlamaParse**（它会自动去除页眉页脚）：

```python
# 如果你是 PDF 且格式复杂，建议替换第一步的数据加载：
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="你的LLAMA_CLOUD_API_KEY", # 需去 cloud.llamaindex.ai 申请
    result_type="markdown"  # 输出为 Markdown 格式，能保留标题结构
)

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
```

这个方案是目前针对书籍索引**性价比和准确度平衡得最好**的方案。你可以直接运行上面的代码进行测试。