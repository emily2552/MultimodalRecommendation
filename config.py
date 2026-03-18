"""
配置文件
包含公用的配置、工具函数和预训练模型
"""

import os
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ============== 设备配置 ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")


# ============== 路径配置 ==============
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ============== 模型配置 ==============
BERT_MODEL_NAME = "bert-base-uncased"
MAX_TEXT_LENGTH = 128
IMAGE_SIZE = 224


# ============== 图像处理 ==============
import torchvision.transforms as transforms

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============== 工具函数 ==============
import requests
from PIL import Image
from io import BytesIO


def download_image(url: str, timeout: int = 10) -> Image.Image:
    """
    下载图片并返回 PIL Image
    
    Args:
        url: 图片 URL
        timeout: 超时时间（秒）
        
    Returns:
        PIL Image 对象
        
    Raises:
        Exception: 下载失败时抛出异常
    """
    if url is None:
        raise ValueError("URL cannot be None")
    
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img


def load_image_tensor(url: str) -> torch.Tensor:
    """
    加载图片并转换为预处理后的 Tensor
    
    Args:
        url: 图片 URL
        
    Returns:
        预处理后的图像张量, 形状 (3, 224, 224)
    """
    try:
        img = download_image(url)
        img_tensor = image_transform(img)
        return img_tensor
    except Exception as e:
        # 下载失败返回全零张量
        return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)


def get_image_placeholder() -> torch.Tensor:
    """
    获取全零占位符图像
    
    Returns:
        全零图像张量, 形状 (3, 224, 224)
    """
    return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)


# ============== 初始化 Tokenizer ==============
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
