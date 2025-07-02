# vector_indexer.py

"""
负责为解析后的信息块（Chunks）创建并管理一个多模态向量索引。

该模块的核心是 `VectorIndexer` 类，它加载文本和视觉嵌入模型，
并将文本、表格和图像的双重表示（文本描述向量和视觉向量）
存入一个持久化的 ChromaDB 向量数据库中。
"""

import logging
import os
import io
from typing import List

import chromadb
from PIL import Image
from sentence_transformers import SentenceTransformer

# 导入我们第一阶段定义的 Chunk 数据结构
from multimodal_parser import Chunk

# --- 配置常量 ---
from config import Config
config = Config()
TEXT_EMBEDDING_MODEL_ID = config.TEXT_EMBEDDING_MODEL_ID
CLIP_MODEL_ID = config.CLIP_MODEL_ID
CHROMA_DB_PATH = config.CHROMA_DB_PATH
CHROMA_COLLECTION_NAME = config.CHROMA_COLLECTION_NAME

# --- 日志配置 (与 parser 保持一致) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VectorIndexer:
    """
    一个用于构建多模态向量索引的类。

    该类封装了加载嵌入模型、连接数据库以及将 Chunk 列表
    转换为多模态向量并存入 ChromaDB 的所有逻辑。
    """

    def __init__(self):
        """初始化 VectorIndexer，加载模型并连接到数据库。"""
        self.text_model: SentenceTransformer
        self.clip_model: SentenceTransformer
        self._load_models()

        logging.info(f"正在连接到 ChromaDB (路径: {self.CHROMA_DB_PATH})...")
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        logging.info(f"成功连接到集合 '{CHROMA_COLLECTION_NAME}'。")

    def _load_models(self) -> None:
        """加载文本嵌入模型和 CLIP 模型。"""
        logging.info(f"正在加载文本嵌入模型: {TEXT_EMBEDDING_MODEL_ID}...")
        self.text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL_ID)
        logging.info("文本嵌入模型加载成功。")

        logging.info(f"正在加载 CLIP 多模态模型: {CLIP_MODEL_ID}...")
        self.clip_model = SentenceTransformer(CLIP_MODEL_ID)
        logging.info("CLIP 模型加载成功。")

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        为给定的 Chunk 列表构建或更新向量索引。

        该方法实现了双重表示逻辑：
        1. 文本/表格/图片描述 -> 文本向量
        2. 图片原始数据 -> 视觉向量

        Args:
            chunks: 从 multimodal_parser.py 生成的 Chunk 对象列表。
        """
        if not chunks:
            logging.warning("没有信息块需要索引。")
            return

        logging.info(f"开始为 {len(chunks)} 个信息块构建索引...")
        
        # 准备批量处理的数据
        text_batch_for_embedding = []
        image_batch_for_embedding = []
        ids_batch = []
        metadatas_batch = []
        documents_batch = []

        for chunk in chunks:
            if chunk.content_type in ["text", "table"]:
                text_batch_for_embedding.append(chunk.content)
                ids_batch.append(chunk.chunk_id)
                metadatas_batch.append({
                    "doc_name": chunk.doc_name,
                    "content_type": chunk.content_type
                })
                documents_batch.append(chunk.content)

            elif chunk.content_type == "image":
                # --- 双重表示的关键逻辑 ---
                # 1. 为图片的“文本描述”准备文本向量
                text_batch_for_embedding.append(chunk.content) # chunk.content 是 LLaVA 描述
                ids_batch.append(f"{chunk.chunk_id}_desc")
                metadatas_batch.append({
                    "doc_name": chunk.doc_name,
                    "content_type": "image_description",
                    "original_image_id": chunk.chunk_id
                })
                documents_batch.append(chunk.content)

                # 2. 为“图片本身”准备视觉向量
                image_bytes = chunk.metadata.get("image_bytes")
                if image_bytes:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    image_batch_for_embedding.append(pil_image)
                    ids_batch.append(f"{chunk.chunk_id}_visual")
                    metadatas_batch.append({
                        "doc_name": chunk.doc_name,
                        "content_type": "image_visual",
                        "original_image_id": chunk.chunk_id,
                        "description": chunk.content # 将描述也存入元数据，便于检索后直接使用
                    })
                    documents_batch.append(f"Visual content of image from {chunk.doc_name}")

        # --- 批量编码和索引 ---
        logging.info(f"正在批量生成 {len(text_batch_for_embedding)} 个文本向量...")
        text_embeddings = self.text_model.encode(
            text_batch_for_embedding, show_progress_bar=True
        )

        logging.info(f"正在批量生成 {len(image_batch_for_embedding)} 个视觉向量...")
        image_embeddings = self.clip_model.encode(
            image_batch_for_embedding, show_progress_bar=True
        )
        
        # 合并所有向量和相关数据
        all_embeddings = list(text_embeddings) + list(image_embeddings)
        
        logging.info(f"正在将总共 {len(ids_batch)} 个条目添加到 ChromaDB 集合中...")
        self.collection.add(
            ids=ids_batch,
            embeddings=all_embeddings,
            documents=documents_batch,
            metadatas=metadatas_batch
        )
        logging.info("索引构建/更新成功！")


if __name__ == '__main__':
    # 这是一个演示如何将阶段一和阶段二串联起来的测试入口
    
    # 动态导入阶段一的模块和函数
    from multimodal_parser import parse_document, initialize_llava_model

    logging.info("--- 开始执行 vector_indexer 模块测试 (串联阶段一) ---")
    
    # --- 阶段一：解析文档 ---
    test_doc_path = "report.docx"
    parsed_chunks = []
    
    if os.path.exists(test_doc_path):
        model_tuple = initialize_llava_model()
        if model_tuple:
            llava_model, llava_processor, device = model_tuple
            parsed_chunks = parse_document(test_doc_path, llava_model, llava_processor, device)
        else:
            logging.error("无法继续，因为 LLaVA 模型加载失败。")
    else:
        logging.warning(f"测试文档 '{test_doc_path}' 不存在。跳过解析和索引。")

    # --- 阶段二：构建索引 ---
    if parsed_chunks:
        try:
            indexer = VectorIndexer()
            indexer.build_index(parsed_chunks)
            
            # 验证索引结果
            count = indexer.collection.count()
            logging.info(f"索引构建完成。当前集合中共有 {count} 个条目。")
            logging.info("可以通过查询 `peek()` 来查看一些示例条目：")
            print(indexer.collection.peek(limit=5))
            
        except Exception as e:
            logging.error(f"在索引过程中发生严重错误: {e}", exc_info=True)