
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import tempfile
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

# 用 langchain 标准组件，彻底不依赖 raglight
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ---------- 网页配置 ----------
st.set_page_config(page_title="我的 RAG 知识库")
st.title("我的私有 RAG 知识库")

# ---------- 加载环境变量 ----------
load_dotenv()
API_KEY = os.getenv("MY_API_KEY")


# ---------- 缓存模型加载（只加载一次） ----------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=API_KEY,
        temperature=0.3,
    )
    return embeddings, llm


embeddings, llm = load_models()


# ---------- 侧边栏：上传并解析 PDF ----------
with st.sidebar:
    st.header("文档管理")
    uploaded_file = st.file_uploader("请上传一个 PDF 文件", type="pdf")

    if uploaded_file is not None:
        if st.button("解析并存入数据库"):
            with st.spinner("正在解析 PDF 并构建向量库..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                # 逐页读取 PDF
                reader = PdfReader(tmp_path)
                documents = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": uploaded_file.name, "page": i + 1},
                        ))

                # 存入向量库（用云端可写的 /tmp 临时目录）
                vector_store = Chroma(
                    collection_name="web_knowledge_base",
                    embedding_function=embeddings,
                    persist_directory="/tmp/chroma_db",
                )
                vector_store.add_documents(documents)

                st.session_state.vector_store = vector_store
                st.success(f"成功解析 {len(documents)} 页内容！你可以开始提问了。")
                os.unlink(tmp_path)


# ---------- 主界面：聊天 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("关于这份文档，你想问什么？"):
    if "vector_store" not in st.session_state:
        st.error("请先在左侧上传 PDF 并点击解析！")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 检索最相关的 4 段内容
        docs = st.session_state.vector_store.similarity_search(prompt, k=4)
        context = "\n\n".join(d.page_content for d in docs)

        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", "你是知识库助手，请仅依据下面的参考资料回答问题。\n参考资料：\n{context}"),
            ("user", "{question}"),
        ])
        chain = prompt_tmpl | llm

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                answer = chain.invoke({"context": context, "question": prompt}).content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})