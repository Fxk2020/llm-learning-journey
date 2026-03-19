import os
# 设置 HuggingFace 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
# 引入重排序器
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
# 使用本地 HuggingFace embedding 模型
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 设置本地 embedding 模型 (支持中文的 BGE 模型)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 1. 加载数据并构建索引
documents = SimpleDirectoryReader("./books/testRag").load_data()
index = VectorStoreIndex.from_documents(documents)

# 2. 初始化 Reranker (使用 BAAI 的 BGE 模型，支持中英文)
# 这会自动下载模型，如果不从 HuggingFace 下载，需配置镜像或离线加载
reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-large",
    top_n=5  # 最终只给大模型看最准的 5 条
)

# 3. 创建查询引擎 (Query Engine)
# 在这里把 reranker 加进去
query_engine = index.as_query_engine(
    similarity_top_k=10, # 初选选出 10 条
    node_postprocessors=[reranker] # 再次精选出 5 条
)

# 4. 提问
response = query_engine.query("日本的老年社会体现在那些方面")
print(response)