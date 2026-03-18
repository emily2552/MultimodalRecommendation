"""
多模态推荐系统数据加载脚本
使用本地 Amazon Reviews 2023 数据集 (All_Beauty 分类)

输出: PyTorch DataLoader
- user_id: 用户ID
- item_id: 商品ID (parent_asin)
- text_raw: 原始评论文本
- image_url: 商品图片URL (从Metadata中获取第一张图)
- rating: 评分 (作为Label)
"""

import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Optional
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer, BertModel

# 初始化全局 tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 自动检测设备：有 GPU 用 GPU，没有则用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

# 将模型加载到对应设备上
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()



def get_text_embedding(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    获取 BERT 最后一层的 [CLS] 向量
    """
    # 1. 切换到评估模式（关闭 Dropout）
    bert_model.eval()

    # 2. 关闭梯度计算（省内存，提速度）
    with torch.no_grad():
        # 同时传入 mask 才是完整的 BERT 调用
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)

    # 取第一个向量 [CLS]，形状为 (batch_size, 768)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding
# 数据目录
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


class AmazonMultimodalDataset(Dataset):
    """
    多模态推荐数据集
    支持文本(评论)和图像(商品图片URL)两种模态
    """
    
    def __init__(
        self, 
        reviews_data: pd.DataFrame, 
        metadata_dict: Dict[str, Dict[str, Any]]
    ):
        """
        初始化数据集
        
        Args:
            reviews_data: Reviews 数据 (DataFrame)
            metadata_dict: Metadata 字典 {parent_asin: metadata_dict}
        """
        self.reviews_data = reviews_data.reset_index(drop=True)
        self.metadata_dict = metadata_dict
        
    def __len__(self) -> int:
        return len(self.reviews_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        Returns:
            包含以下字段的字典:
            - user_id: 用户ID
            - item_id: 商品ID (parent_asin)
            - text_raw: 原始评论文本
            - image_url: 商品图片URL (若无则为None)
            - rating: 评分 (float)
        """
        row = self.reviews_data.iloc[idx]
        
        # 获取用户ID和商品ID
        user_id = row['user_id']
        item_id = row['parent_asin']
        
        # 获取评论文本 (text 字段)
        text_raw = str(row.get('text', '')) if pd.notna(row.get('text', '')) else ''
        
        # 从 Metadata 获取图片URL
        image_url = None
        if item_id in self.metadata_dict:
            meta = self.metadata_dict[item_id]
            images = meta.get('images', [])
            if images and isinstance(images, list) and len(images) > 0:
                # 取第一张图片
                first_image = images[0]
                if isinstance(first_image, dict):
                    # 优先获取 hi_res，然后是 large，最后是 thumb
                    for key in ['hi_res', 'large', 'thumb']:
                        if key in first_image and first_image[key]:
                            image_url = first_image[key]
                            break
        
        # 获取评分
        rating = float(row.get('rating', 0.0))
        
        return {
            'user_id': user_id,
            'item_id': item_id,
            'text_raw': text_raw,
            'image_url': image_url,
            'rating': rating
        }


def collate_fn(batch):
    """
    自定义 collate 函数，处理 batch 中的数据
    
    Args:
        batch: 样本列表
        
    Returns:
        打包后的 batch 字典
    """
    # 提取 user_id 和 item_id 列表
    user_ids = [item['user_id'] for item in batch]
    item_ids = [item['item_id'] for item in batch]
    
    # 提取文本列表
    text_list = [item['text_raw'] for item in batch]
    
    # 使用 tokenizer 处理文本
    encoded = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # 获取其他字段
    image_urls = [item['image_url'] for item in batch]
    ratings = torch.tensor([item['rating'] for item in batch], dtype=torch.float32)
    
    return {
        'user_id': user_ids,
        'item_id': item_ids,
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'image_url': image_urls,
        'rating': ratings
   }


def load_jsonl(filepath: str, max_lines: Optional[int] = None) -> list:
    """
    逐行加载 JSONL 文件
    
    Args:
        filepath: 文件路径
        max_lines: 最大行数 (None 表示全部)
        
    Returns:
        字典列表
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_local_amazon_data(
    reviews_file: str = "All_Beauty.jsonl",
    metadata_file: str = "meta_All_Beauty.jsonl",
    max_samples: Optional[int] = None
) -> tuple:
    """
    加载本地的 Amazon Reviews 2023 数据集
    
    Args:
        reviews_file: Reviews 数据文件名
        metadata_file: Metadata 数据文件名
        max_samples: 最大样本数 (默认: None, 加载全部)
        
    Returns:
        (reviews_df, metadata_dict)
    """
    reviews_path = os.path.join(DATA_DIR, reviews_file)
    metadata_path = os.path.join(DATA_DIR, metadata_file)
    
    print(f"正在加载 Reviews 数据: {reviews_path}")
    reviews_list = load_jsonl(reviews_path, max_lines=max_samples)
    reviews_df = pd.DataFrame(reviews_list)
    print(f"  Reviews 数据加载完成，共 {len(reviews_df)} 条")
    
    print(f"正在加载 Metadata 数据: {metadata_path}")
    metadata_list = load_jsonl(metadata_path)
    metadata_dict = {item['parent_asin']: item for item in metadata_list}
    print(f"  Metadata 加载完成，共 {len(metadata_dict)} 个商品")
    
    return reviews_df, metadata_dict


def create_dataloader(
    reviews_df: pd.DataFrame,
    metadata_dict: Dict[str, Dict[str, Any]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    创建 PyTorch DataLoader
    
    Args:
        reviews_df: Reviews 数据 DataFrame
        metadata_dict: Metadata 字典
        batch_size: Batch 大小
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        
    Returns:
        PyTorch DataLoader
    """
    dataset = AmazonMultimodalDataset(reviews_df, metadata_dict)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader


def preprocess_and_create_dataloader(
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    一站式函数：加载数据并创建 DataLoader
    
    Args:
        batch_size: Batch 大小
        max_samples: 最大样本数
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        
    Returns:
        PyTorch DataLoader
    """
    # 加载数据
    reviews_df, metadata_dict = load_local_amazon_data(
        max_samples=max_samples
    )
    
    # 创建 DataLoader
    dataloader = create_dataloader(
        reviews_df=reviews_df,
        metadata_dict=metadata_dict,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader


# 示例用法
if __name__ == "__main__":
    # 加载100条数据作为示例
    dataloader = preprocess_and_create_dataloader(
        batch_size=4,
        max_samples=100,
        shuffle=True
    )
    
    print(f"\nDataLoader 创建成功！共有 {len(dataloader)} 个 batches")
    
    # 测试读取一个 batch
    for batch in dataloader:
        print("\n=== Batch 示例 ===")
        print(f"user_id: {batch['user_id']}")
        print(f"item_id: {batch['item_id']}")
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"attention_mask shape: {batch['attention_mask'].shape}")

        # 修改调用方式，传入 mask
        cls_embedding = get_text_embedding(batch['input_ids'], batch['attention_mask'])
        print(f"[CLS] embedding shape: {cls_embedding.shape}")
        print(f"image_url: {batch['image_url']}")
        print(f"rating: {batch['rating']}")
        print(f"rating shape: {batch['rating'].shape}")
        break
