import os


class Config:
    """
    配置类，用于存储模型和处理器的配置信息。

    Attributes:
        LLAVA_MODEL_ID: LLAVA模型的ID
        TEXT_EMBEDDING_MODEL_ID: 文本嵌入模型的ID
        CLIP_MODEL_ID: CLIP模型的ID
        CHROMA_DB_PATH: ChromaDB的持久化存储路径
        CHROMA_COLLECTION_NAME: ChromaDB集合的名称
    """
    
    LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    TEXT_EMBEDDING_MODEL_ID = "BAAI/bge-base-en-v1.5"
    CLIP_MODEL_ID = "sentence-transformers/clip-ViT-B-32"
    CHROMA_DB_PATH = "./chroma_db"  # 持久化存储路径
    CHROMA_COLLECTION_NAME = "multimodal_collection_v2"
    LLM_MODEL_ID = "gpt-4o"
    OPENAI_API_KEY = ""