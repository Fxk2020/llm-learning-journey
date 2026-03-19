
from llama_cloud_services import LlamaCloudIndex
# pip install llama-cloud-services

index = LlamaCloudIndex(
  name="yiriweijian",
  project_name="Default",
  organization_id="c342d82f-5132-4e60-9e34-04261ee0e43a",
  api_key="llx-pArjJJzWAvb13TCLLeoRJjqAICnF17sqntY15d9wsUYuVLjS",
)
query = "日本的老年社会体现在那些方面"
nodes = index.as_retriever().retrieve(query)
response = index.as_query_engine().query(query)
