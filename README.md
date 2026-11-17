# My RAGLight Application

基于开源框架 RAGLight 进行二次开发的检索增强生成（RAG）应用。

## 💡 项目亮点 (二次开发)
1. **混合架构设计**：由于部分国内大模型 API 对 LangChain 接口兼容性不佳，本项目将原生 Embedding 模块替换为本地开源的 `BAAI/bge-small-zh-v1.5` 模型，实现离线向量化。
2. **云端推理**：使用阿里云百炼（通义千问 qwen-plus）作为 LLM 进行最终生成。
3. **数据持久化**：结合 Chroma 向量数据库，实现本地知识库的存储与检索。

## 🛠️ 快速开始
1. 安装依赖：`pip install raglight langchain-huggingface langchain-chroma`
2. 在 `main.py` 中配置你的百炼 API Key。
3. 运行程序：`python main.py`