"""
多模态推荐系统数据加载脚本
使用 Amazon Reviews 2023 数据集 (All_Beauty 分类)

输出: PyTorch DataLoader
- user_id: 用户ID
- item_id: 商品ID (parent_asin)
- text_raw: 原始评论文本
- image_url: 商品图片URL (从Metadata中获取第一张图)
- rating: 评分 (作为Label)
"""

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Optional
import pandas as pd

# 忽略 datasets 库的不必要输出
datasets.logging.set_verbosity_error()


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
        item_id = row['parent_asin']  # 使用 parent_asin 作为 item_id
        
        # 获取评论文本 (text 字段)
        text_raw = row.get('text', '') or ''
        
        # 从 Metadata 获取图片URL
        image_url = None
        if item_id in self.metadata_dict:
            meta = self.metadata_dict[item_id]
            images = meta.get('images', {})
            if images and isinstance(images, dict):
                # 优先获取 hi_res，然后是 large，最后是 thumb
                for key in ['hi_res', 'large', 'thumb']:
                    if key in images and images[key]:
                        # 取第一张图片
                        url = images[key][0] if isinstance(images[key], list) else images[key]
                        if url is not None:
                            image_url = url
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
    return {
        'user_id': [item['user_id'] for item in batch],
        'item_id': [item['item_id'] for item in batch],
        'text_raw': [item['text_raw'] for item in batch],
        'image_url': [item['image_url'] for item in batch],
        'rating': torch.tensor([item['rating'] for item in batch], dtype=torch.float32)
    }


def load_amazon_reviews_2023(
    category: str = "All_Beauty",
    max_samples: Optional[int] = None,
    sample_ratio: float = 1.0
) -> tuple:
    """
    加载 Amazon Reviews 2023 数据集的 Reviews 和 Item Metadata
    
    Args:
        category: 商品分类 (默认: All_Beauty)
        max_samples: 最大样本数 (默认: None, 加载全部)
        sample_ratio: 采样比例 (0-1之间)
        
    Returns:
        (reviews_df, metadata_dict)
    """
    print(f"正在加载 {category} 分类的 Reviews 数据...")
    
    # 加载 Reviews 数据
    reviews_dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        trust_remote_code=True
    )
    reviews_full = reviews_dataset["full"]
    
    # 转换为 DataFrame
    reviews_df = reviews_full.to_pandas()
    print(f"  Reviews 数据加载完成，共 {len(reviews_df)} 条")
    
    # 根据 max_samples 截取
    if max_samples is not None and max_samples < len(reviews_df):
        reviews_df = reviews_df.head(max_samples)
        print(f"  截取前 {max_samples} 条数据")
    
    # 根据 sample_ratio 采样
    if sample_ratio < 1.0:
        reviews_df = reviews_df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        print(f"  采样比例 {sample_ratio}，剩余 {len(reviews_df)} 条数据")
    
    print(f"正在加载 {category} 分类的 Item Metadata...")
    
    # 加载 Item Metadata
    metadata_dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{category}",
        split="full",
        trust_remote_code=True
    )
    
    # 转换为字典 {parent_asin: metadata}
    metadata_list = metadata_dataset.to_list()
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
    category: str = "All_Beauty",
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    sample_ratio: float = 1.0,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    一站式函数：加载数据并创建 DataLoader
    
    Args:
        category: 商品分类
        batch_size: Batch 大小
        max_samples: 最大样本数
        sample_ratio: 采样比例
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        
    Returns:
        PyTorch DataLoader
    """
    # 加载数据
    reviews_df, metadata_dict = load_amazon_reviews_2023(
        category=category,
        max_samples=max_samples,
        sample_ratio=sample_ratio
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
    # 加载少量数据作为示例 (100条，采样1%)
    dataloader = preprocess_and_create_dataloader(
        category="All_Beauty",
        batch_size=4,
        max_samples=100,
        sample_ratio=0.01,
        shuffle=True
    )
    
    print(f"\nDataLoader 创建成功！共有 {len(dataloader)} 个 batches")
    
    # 测试读取一个 batch
    for batch in dataloader:
        print("\n=== Batch 示例 ===")
        print(f"user_id: {batch['user_id']}")
        print(f"item_id: {batch['item_id']}")
        print(f"text_raw (前100字符): {[t[:100] + '...' if len(t) > 100 else t for t in batch['text_raw']]}")
        print(f"image_url: {batch['image_url']}")
        print(f"rating: {batch['rating']}")
        print(f"rating shape: {batch['rating'].shape}")
        break
