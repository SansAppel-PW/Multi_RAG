# rag_retriever.py

"""
负责接收用户查询，从多模态索引中检索相关上下文，并调用LLM生成答案。

该模块的核心是 RAGRetriever 类，它实现了多路并行检索、结果融合、
提示词构建和与 OpenAI API 交互的完整 RAG 流程。
"""
import logging
import os
from typing import Any, Dict, List

import chromadb
import openai
from sentence_transformers import SentenceTransformer

# --- 配置常量 ---
# 从环境变量中读取 OpenAI API 密钥
from config import Config
config = Config()
OPENAI_API_KEY = os.getenv(config.OPENAI_API_KEY)

LLM_MODEL_ID = config.LLM_MODEL_ID

# 与 vector_indexer.py 中保持一致的配置
TEXT_EMBEDDING_MODEL_ID = config.TEXT_EMBEDDING_MODEL_ID
CLIP_MODEL_ID = config.CLIP_MODEL_ID
CHROMA_DB_PATH = config.CHROMA_DB_PATH
CHROMA_COLLECTION_NAME = config.CHROMA_COLLECTION_NAME

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGRetriever:
    """
    一个完整的 RAG (检索增强生成) 系统。

    它封装了从接收问题到生成答案的全过程，包括多模态检索、
    上下文融合和调用 LLM。
    """

    def __init__(self):
        """初始化 RAGRetriever，加载所需模型并连接到服务。"""
        if not OPENAI_API_KEY:
            raise ValueError("环境变量 'OPENAI_API_KEY' 未设置。")
        
        self.text_model: SentenceTransformer
        self.clip_model: SentenceTransformer
        self._load_models()

        logging.info(f"正在连接到已存在的 ChromaDB 集合 '{CHROMA_COLLECTION_NAME}'...")
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_collection(name=CHROMA_COLLECTION_NAME)
        
        self.llm_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logging.info("RAGRetriever 初始化成功。")

    def _load_models(self) -> None:
        """加载文本嵌入模型和 CLIP 模型。"""
        logging.info(f"正在加载文本嵌入模型: {TEXT_EMBEDDING_MODEL_ID}...")
        self.text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL_ID)
        logging.info("文本嵌入模型加载成功。")

        logging.info(f"正在加载 CLIP 多模态模型: {CLIP_MODEL_ID}...")
        self.clip_model = SentenceTransformer(CLIP_MODEL_ID)
        logging.info("CLIP 模型加载成功。")

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        执行多模态检索并融合结果。

        Args:
            question: 用户提出的问题。
            top_k: 从每个检索器中召回的文档数量。

        Returns:
            一个融合、去重后的上下文块列表。
        """
        logging.info(f"开始为问题执行多模态检索: '{question}'")
        
        # --- 1. 并行查询 ---
        # 查询 A: 文本语义检索 (文本、表格、图片描述)
        q_text_emb = self.text_model.encode(question).tolist()
        text_results = self.collection.query(
            query_embeddings=[q_text_emb],
            n_results=top_k,
            where={"content_type": {"$in": ["text", "table", "image_description"]}}
        )

        # 查询 B: 文-图视觉检索
        q_clip_emb = self.clip_model.encode(question).tolist()
        image_results = self.collection.query(
            query_embeddings=[q_clip_emb],
            n_results=top_k,
            where={"content_type": "image_visual"}
        )

        # --- 2. 结果融合与去重 ---
        fused_results: Dict[str, Dict[str, Any]] = {}
        
        # 统一处理函数，避免代码重复
        def process_query_results(results: Dict[str, Any]):
            ids = results.get('ids', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            documents = results.get('documents', [[]])[0]
            
            for i, doc_id in enumerate(ids):
                # 对于图片，使用 original_image_id 作为去重的唯一标识
                unique_id = metadatas[i].get("original_image_id", doc_id)
                if unique_id not in fused_results:
                     # 关键：对于视觉检索结果，我们使用其元数据中存储的“文字描述”作为上下文
                    content_to_use = metadatas[i].get("description", documents[i])
                    fused_results[unique_id] = {
                        "content": content_to_use,
                        "metadata": metadatas[i]
                    }

        process_query_results(text_results)
        process_query_results(image_results)

        logging.info(f"检索到 {len(text_results.get('ids', [[]])[0])} 个文本相关块和 "
                     f"{len(image_results.get('ids', [[]])[0])} 个视觉相关块。")
        logging.info(f"融合去重后，得到 {len(fused_results)} 个唯一的上下文块。")
        
        return list(fused_results.values())

    def _format_prompt(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        构建最终发送给 LLM 的、结构化的提示词。

        Args:
            question: 用户的原始问题。
            context_chunks: 经过检索和融合的上下文块列表。

        Returns:
            一个完整的、包含指令和上下文的提示词字符串。
        """
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            doc_name = chunk['metadata'].get('doc_name', 'N/A')
            content_type = chunk['metadata'].get('content_type', 'N/A')
            
            context_str += f"--- Context [{i+1}] (Source: {doc_name}, Type: {content_type}) ---\n"
            context_str += chunk['content'] + "\n\n"
        
        prompt = f"""You are an expert Q&A assistant. Your answer must be based solely on the provided context.
If the context does not contain the answer, state that you cannot answer based on the provided information.
Do not make up information. Cite the source of your answer using the format [Source X].

--- CONTEXT ---
{context_str}
--- QUESTION ---
{question}

--- ANSWER ---
"""
        return prompt

    def query(self, question: str) -> str:
        """
        执行完整的 RAG 流程：检索 -> 构建提示词 -> 生成答案。

        Args:
            question: 用户提出的问题。

        Returns:
            由 LLM 生成的最终答案。
        """
        context = self.retrieve(question)
        if not context:
            return "I could not find any relevant information in the knowledge base to answer this question."

        prompt = self._format_prompt(question, context)
        
        logging.info("正在向 OpenAI API 发送请求以生成答案...")
        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 偏向于确定性的回答
            )
            final_answer = response.choices[0].message.content
            logging.info("成功从 API 获取答案。")
            return final_answer
        except Exception as e:
            logging.error(f"调用 OpenAI API 时出错: {e}", exc_info=True)
            return "An error occurred while communicating with the language model."


if __name__ == '__main__':
    logging.info("--- 开始执行 rag_retriever 模块测试 ---")
    
    try:
        retriever = RAGRetriever()
        
        # --- 定义一组测试问题 ---
        # 这个问题可能匹配表格或文本
        test_question_1 = "What are the key financial figures mentioned in the report?" 
        # 这个问题可能匹配流程图或架构图
        test_question_2 = "Can you describe the user authentication flow?"
        # 这个问题可能需要结合文本和图片信息
        test_question_3 = "Summarize the project's main outcome and show the final architecture diagram."

        test_questions = [test_question_1, test_question_2, test_question_3]

        for q in test_questions:
            print("\n" + "="*50)
            print(f"QUESTION: {q}")
            print("="*50)
            answer = retriever.query(q)
            print(f"ANSWER:\n{answer}")

    except ValueError as e:
        logging.error(e)
    except Exception as e:
        logging.critical(f"RAGRetriever 初始化或查询过程中发生严重错误: {e}", exc_info=True)