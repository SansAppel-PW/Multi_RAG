# multimodal_parser.py

"""
负责深度解析包含文本、表格和图像的 Word (.docx) 文档。

该模块的核心功能是 `parse_document` 函数，它将一个 .docx 文件
转换为一个结构化的信息块（Chunk）列表。对于图像，它会利用
一个多模态大模型生成详细的文本描述，并保留原始图像数据以供后续处理。
"""

import io
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from docx import Document
from docx.table import Table as DocxTable
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# --- 配置常量 ---
from config import Config
config = Config()
LLAVA_MODEL_ID = config.LLAVA_MODEL_ID

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 核心数据结构 ---
@dataclass
class Chunk:
    """标准化的信息块数据结构。

    Attributes:
        chunk_id: 唯一的块标识符。
        doc_name: 来源文档的名称。
        content: 块的主要内容。对于文本/表格是其字符串，对于图像是其生成的描述。
        content_type: 块的类型 ("text", "table", "image")。
        metadata: 包含附加信息的字典，例如图像的原始二进制数据。
    """
    doc_name: str
    content: str
    content_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

# --- 模型加载与初始化 ---
def initialize_llava_model() -> Optional[Tuple[LlavaForConditionalGeneration, AutoProcessor, str]]:
    """
    加载并初始化 LLaVA 多模态模型。

    Returns:
        一个元组，包含加载的模型、处理器和设备("cuda:0"或"cpu")。
        如果加载失败，则返回 None。
    """
    device = "cpu"
    torch_dtype = torch.float32
    load_in_4bit = False

    if torch.cuda.is_available():
        logging.info("检测到 CUDA，将在 GPU 上以 4-bit 模式加载 LLaVA 模型。")
        device = "cuda:0"
        torch_dtype = torch.float16
        load_in_4bit = True
    else:
        logging.warning("未检测到 CUDA。LLaVA 模型将在 CPU 上运行，速度会非常慢。")

    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            load_in_4bit=load_in_4bit,
        )
        processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
        logging.info("LLaVA 模型加载成功。")
        return model, processor, device
    except Exception as e:
        logging.error(f"加载 LLaVA 模型失败: {e}", exc_info=True)
        return None, None, None

# --- 核心功能函数 ---
def _describe_image(image_bytes: bytes, model: LlavaForConditionalGeneration, processor: AutoProcessor, device: str) -> str:
    """使用 LLaVA 模型为给定的图像二进制数据生成描述。"""
    try:
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prompt = "USER: <image>\nDescribe this image in detail. Focus on any text, data, or structural elements present."
        inputs = processor(prompt, images=raw_image, return_tensors="pt").to(device, model.dtype)
        output = model.generate(**inputs, max_new_tokens=250, do_sample=False)
        description = processor.decode(output[0][2:], skip_special_tokens=True)
        return description.strip()
    except Exception as e:
        logging.error(f"图像描述生成过程中发生错误: {e}", exc_info=True)
        return "Failed to generate image description."

def _table_to_markdown(table: DocxTable) -> str:
    """将 docx.table 对象转换为 Markdown 格式。"""
    md_rows = []
    header = [cell.text.strip().replace("\n", " ") for cell in table.rows[0].cells]
    md_rows.append("| " + " | ".join(header) + " |")
    md_rows.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in table.rows[1:]:
        row_data = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
        md_rows.append("| " + " | ".join(row_data) + " |")
    return "\n".join(md_rows)

def parse_document(file_path: str, model: LlavaForConditionalGeneration, processor: AutoProcessor, device: str) -> List[Chunk]:
    """
    解析单个 Word 文档，提取所有内容并将其转换为 Chunk 列表。

    Args:
        file_path: Word 文档的路径。
        model: 已初始化的 LLaVA 模型。
        processor: 已初始化的 LLaVA 处理器。
        device: 模型所在的设备 ("cuda:0" 或 "cpu")。

    Returns:
        一个包含所有从文档中提取出的信息块（Chunk）的列表。
    """
    logging.info(f"开始解析文档: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"文件不存在: {file_path}")
        return []

    doc = Document(file_path)
    doc_name = os.path.basename(file_path)
    chunks: List[Chunk] = []
    image_rels = {r.rId: r.target_part.blob for r in doc.part.rels.values() if "image" in r.target_ref}
    text_buffer = ""

    for element in doc.element.body:
        if element.tag.endswith('p'):  # 段落
            para_text = element.text
            r_embeds = element.xpath('.//a:blip/@r:embed')
            if r_embeds and r_embeds[0] in image_rels:
                if text_buffer.strip():
                    chunks.append(Chunk(doc_name=doc_name, content=text_buffer.strip(), content_type="text"))
                    text_buffer = ""
                
                image_bytes = image_rels[r_embeds[0]]
                logging.info(f"  -> 在文档 '{doc_name}' 中发现图像，开始生成描述...")
                description = _describe_image(image_bytes, model, processor, device)
                chunks.append(Chunk(
                    doc_name=doc_name,
                    content=description,
                    content_type="image",
                    metadata={"image_bytes": image_bytes}
                ))
            if para_text.strip():
                text_buffer += para_text + "\n"
        
        elif element.tag.endswith('tbl'):  # 表格
            if text_buffer.strip():
                chunks.append(Chunk(doc_name=doc_name, content=text_buffer.strip(), content_type="text"))
                text_buffer = ""
            
            logging.info(f"  -> 在文档 '{doc_name}' 中发现表格，转换为 Markdown...")
            table_obj = next((t for t in doc.tables if t._element is element), None)
            if table_obj:
                md_table = _table_to_markdown(table_obj)
                chunks.append(Chunk(doc_name=doc_name, content=md_table, content_type="table"))

    if text_buffer.strip():
        chunks.append(Chunk(doc_name=doc_name, content=text_buffer.strip(), content_type="text"))

    logging.info(f"文档 '{doc_name}' 解析完成，共生成 {len(chunks)} 个信息块。")
    return chunks


if __name__ == '__main__':
    logging.info("--- 开始执行 multimodal_parser 模块测试 ---")
    
    # 初始化模型
    model_tuple = initialize_llava_model()
    
    if model_tuple:
        llava_model, llava_processor, device = model_tuple
        
        # 定义测试文档路径
        test_doc_path = "report.docx"
        
        if os.path.exists(test_doc_path):
            # 执行解析
            parsed_chunks = parse_document(test_doc_path, llava_model, llava_processor, device)
            
            # 打印结果预览
            logging.info("--- 解析结果预览 ---")
            for i, chunk in enumerate(parsed_chunks):
                print(f"\n--- Chunk {i+1} (ID: {chunk.chunk_id}) ---")
                print(f"  类型: {chunk.content_type}")
                if chunk.content_type == "image":
                    print(f"  元数据: 包含 image_bytes (大小: {len(chunk.metadata.get('image_bytes', b''))} bytes)")
                print(f"  内容预览: {chunk.content[:200].replace(chr(10), ' ')}...")
        else:
            logging.warning(f"测试文档 '{test_doc_path}' 不存在。请在当前目录下放置一个 .docx 文件进行测试。")
    else:
        logging.error("无法执行解析，因为 LLaVA 模型未能成功加载。")