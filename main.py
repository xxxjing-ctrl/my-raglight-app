import os
from dotenv import load_dotenv
from raglight import RAGPipeline
from raglight.integrations.chroma import ChromaVectorStore
from raglight.integrations.openai import OpenAILLM
from raglight.models import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

# 加载 .env 文件中的环境变量（保护 API Key 不泄露）
load_dotenv()
API_KEY = os.getenv("MY_API_KEY")

print("1. 加载本地 Embedding 模型...")
# 使用本地开源轻量级模型，强制离线加载以避免网络延迟
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"local_files_only": True}
)

print("2. 初始化大语言模型 (通义千问)...")
# 通过阿里云百炼的 OpenAI 兼容接口调用云端大模型
llm = OpenAILLM(
    model_name="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=API_KEY,
)

print("3. 解析本地 PDF 文档...")
# 使用 pypdf 直接提取文本，替代已被官方标记退役的社区加载器
reader = PdfReader("data/test.pdf")
documents = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text.strip():
        # 将提取的文本封装为 RAGLight 的 Document 对象
        # metadata 用于记录数据来源，在后续检索溯源时非常重要
        documents.append(
            Document(
                page_content=text,
                metadata={"source": "test.pdf", "page": i + 1}
            )
        )
print(f"成功解析并加载了 {len(documents)} 页有效文本。")

print("4. 构建本地向量数据库...")
# 将文本向量化并存入 Chroma 数据库，持久化保存到本地磁盘
vector_store = ChromaVectorStore(
    collection_name="my_knowledge_base",
    embeddings_model=embeddings,
    persist_directory="./chroma_db",
)
vector_store.add_documents(documents)

print("5. 运行 RAG 问答流程...")
# 将各个组件组装成完整的 RAG 流水线
pipeline = RAGPipeline(
    llm=llm,
    embeddings=embeddings,
    vector_store=vector_store,
)

# 发起检索提问
question = "请用一句话总结这份文档的核心内容。"
response = pipeline.generate(question)

print("\n--- 大模型回答 ---")
print(response.answer)