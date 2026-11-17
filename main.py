import os
from dotenv import load_dotenv  
from raglight import RAGPipeline
from raglight.integrations.chroma import ChromaVectorStore
from raglight.integrations.openai import OpenAILLM
from raglight.models import Document
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
API_KEY = os.getenv("MY_API_KEY")  


text_content = "这是我自己独立开发的第一个 RAG 项目。我使用了本地的 BGE 模型进行向量化，并结合了阿里云的通义千问。"


print("1. 正在加载本地 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"local_files_only": True}
)

print("2. 正在连接云端大模型...")
llm = OpenAILLM(
    model_name="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=API_KEY,
)

print("3. 正在构建向量数据库（注意，这次会保存在本地）...")
vector_store = ChromaVectorStore(
    collection_name="my_knowledge_base",
    embeddings_model=embeddings,
    persist_directory="./chroma_db", 
)
vector_store.add_documents([Document(page_content=text_content, metadata={"source": "my_test_data"})])

print("4. 组装系统并提问...")
pipeline = RAGPipeline(
    llm=llm,
    embeddings=embeddings,
    vector_store=vector_store,
)

response = pipeline.generate("我使用了什么模型？")
print("\n--- 大模型回答 ---")
print(response.answer)