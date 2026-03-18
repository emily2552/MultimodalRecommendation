"""
模型模块
纯粹的多模态推荐模型，只接受 Tensor 输入
"""

import torch
import torch.nn as nn
from transformers import BertModel
import torchvision.models as models

from config import device, BERT_MODEL_NAME


class MultiModalRecModel(nn.Module):
    """
    多模态推荐模型
    
    结构：
    - 文本分支: BERT 提取特征 (768维) -> 投影到 256维
    - 图像分支: ResNet50 提取特征 (2048维) -> 投影到 256维
    - 拼接得到 512 维
    - 两个全连接层输出预测评分
    
    输入：
    - input_ids: (batch_size, seq_len)
    - attention_mask: (batch_size, seq_len)
    - image_tensors: (batch_size, 3, 224, 224)
    
    输出：
    - 预测评分: (batch_size, 1)
    """
    
    def __init__(self, hidden_dim: int = 256):
        super(MultiModalRecModel, self).__init__()
        
        # 加载预训练 BERT
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        
        # 加载预训练 ResNet50，移除分类层
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        # 文本特征投影层: 768 -> 256
        self.text_projection = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 图像特征投影层: 2048 -> 256
        self.image_projection = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 拼接后全连接: 512 -> 256 -> 1
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: BERT 输入 IDs, 形状 (batch_size, seq_len)
            attention_mask: BERT attention mask, 形状 (batch_size, seq_len)
            image_tensors: 图像张量, 形状 (batch_size, 3, 224, 224)
            
        Returns:
            预测评分, 形状 (batch_size, 1)
        """
        # 1. 文本特征提取
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        
        # 2. 图像特征提取
        image_features = self.resnet(image_tensors)  # (batch, 2048, 1, 1)
        image_features = image_features.squeeze(-1).squeeze(-1)  # (batch, 2048)
        
        # 3. 特征投影
        text_embed = self.text_projection(text_features)   # (batch, 256)
        image_embed = self.image_projection(image_features) # (batch, 256)
        
        # 4. 特征拼接
        combined = torch.cat([text_embed, image_embed], dim=1)  # (batch, 512)
        
        # 5. 全连接层
        x = self.fc1(combined)   # (batch, 256)
        output = self.fc2(x)     # (batch, 1)
        
        return output
    
    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """提取文本特征"""
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return bert_outputs.last_hidden_state[:, 0, :]
    
    def get_image_features(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """提取图像特征"""
        features = self.resnet(image_tensors)
        return features.squeeze(-1).squeeze(-1)


if __name__ == "__main__":
    # 测试代码
    from dataset import preprocess_and_create_dataloader
    
    # 创建模型
    model = MultiModalRecModel()
    model = model.to(device)
    model.eval()
    
    print("模型创建成功!")
    print(f"模型结构:\n{model}")
    
    # 加载数据
    dataloader = preprocess_and_create_dataloader(
        batch_size=4,
        max_samples=100,
        shuffle=True
    )
    
    print(f"\nDataLoader 创建成功！共有 {len(dataloader)} 个 batches")
    
    # 测试一个 batch
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_tensors = batch['image_tensors'].to(device)
        
        print(f"\n输入数据:")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  attention_mask shape: {attention_mask.shape}")
        print(f"  image_tensors shape: {image_tensors.shape}")
        
        # 前向传播
        with torch.no_grad():
            predictions = model(input_ids, attention_mask, image_tensors)
        
        print(f"\n输出:")
        print(f"  预测评分 shape: {predictions.shape}")
        print(f"  预测评分: {predictions.squeeze()}")
        break
