import os
import streamlit as st
from dotenv import load_dotenv
from raglight import RAGPipeline
from raglight.integrations.chroma import ChromaVectorStore
from raglight.integrations.openai import OpenAILLM
from raglight.models import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
import tempfile

# 网页配置
st.set_page_config(page_title="我的 RAG 知识库")
st.title("我的私有 RAG 知识库")

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("MY_API_KEY")

# 缓存模型加载，避免每次对话都重新加载模型
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"local_files_only": True}
    )
    llm = OpenAILLM(
        model_name="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=API_KEY,
    )
    return embeddings, llm

embeddings, llm = load_models()

# 侧边栏：上传 PDF
with st.sidebar:
    st.header("文档管理")
    uploaded_file = st.file_uploader("请上传一个 PDF 文件", type="pdf")
    
    if uploaded_file is not None:
        if st.button("解析并存入数据库"):
            with st.spinner("正在解析 PDF 并构建向量库..."):
                # 把上传的文件存到本地临时目录
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # 读取 PDF
                reader = PdfReader(tmp_path)
                documents = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": uploaded_file.name, "page": i + 1}
                        ))
                
                # 存入向量库
                vector_store = ChromaVectorStore(
                    collection_name="web_knowledge_base",
                    embeddings_model=embeddings,
                    persist_directory="./chroma_db",
                )
                vector_store.add_documents(documents)
                
                # 把组件存到 session_state 中，供聊天界面使用
                st.session_state.pipeline = RAGPipeline(
                    llm=llm,
                    embeddings=embeddings,
                    vector_store=vector_store,
                )
                st.success(f"成功解析 {len(documents)} 页内容！你可以开始提问了。")
                os.unlink(tmp_path) # 清理临时文件

# 主界面：聊天对话框
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("关于这份文档，你想问什么？"):
    if "pipeline" not in st.session_state:
        st.error("请先在左侧上传 PDF 并点击解析！")
    else:
        # 显示用户提问
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = st.session_state.pipeline.generate(prompt)
                st.markdown(response.answer)
                st.session_state.messages.append({"role": "assistant", "content": response.answer})