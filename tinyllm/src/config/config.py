"""
模型配置模块
定义所有超参数和配置选项
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型配置类 - 所有超参数的集中管理"""
    
    # === 模型架构参数 ===
    dim: int = 288                    # 模型隐藏维度
    n_layers: int = 6                 # Transformer层数
    n_heads: int = 6                  # 注意力头数
    n_kv_heads: Optional[int] = None  # K/V头数（用于GQA）
    vocab_size: int = 32000           # 词汇表大小
    max_seq_len: int = 256            # 最大序列长度
    
    # === MLP参数 ===
    hidden_dim: Optional[int] = None  # MLP隐藏层维度
    multiple_of: int = 32             # 隐藏层维度的倍数
    
    # === 归一化参数 ===
    norm_eps: float = 1e-5            # 归一化epsilon
    
    # === 正则化参数 ===
    dropout: float = 0.0              # Dropout概率
    
    # === 旋转位置编码参数 ===
    rope_theta: float = 10000.0       # RoPE基础频率
    
    def __post_init__(self):
        """配置验证和自动设置"""
        # 如果没有指定KV头数，使用默认值
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
            
        # 验证头数配置
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        
        # 自动设置MLP隐藏层维度
        if self.hidden_dim is None:
            self.hidden_dim = 4 * self.dim
            self.hidden_dim = int(2 * self.hidden_dim / 3)
            self.hidden_dim = self.multiple_of * ((self.hidden_dim + self.multiple_of - 1) // self.multiple_of)
    
    def get_model_size(self) -> int:
        """估算模型参数量"""
        # 词嵌入
        embedding_params = self.vocab_size * self.dim
        
        # 每层的参数量
        # 注意力层：Q, K, V, O
        attn_params = self.dim * (self.dim + 2 * self.n_kv_heads * self.dim // self.n_heads + self.dim)
        # MLP层：3个线性层
        mlp_params = self.dim * self.hidden_dim + self.hidden_dim * self.dim + self.dim * self.hidden_dim
        # 归一化层
        norm_params = 2 * self.dim
        
        layer_params = attn_params + mlp_params + norm_params
        total_params = embedding_params + self.n_layers * layer_params + self.dim  # 最后的norm
        
        return total_params
    
    def display_config(self):
        """显示配置信息"""
        print("=== 模型配置 ===")
        print(f"模型维度: {self.dim}")
        print(f"层数: {self.n_layers}")
        print(f"注意力头数: {self.n_heads}")
        print(f"KV头数: {self.n_kv_heads}")
        print(f"词汇表大小: {self.vocab_size}")
        print(f"最大序列长度: {self.max_seq_len}")
        print(f"MLP隐藏层维度: {self.hidden_dim}")
        print(f"Dropout: {self.dropout}")
        print(f"估算参数量: {self.get_model_size():,}")
        print("=" * 20)
