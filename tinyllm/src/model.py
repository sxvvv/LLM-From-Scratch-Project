import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from config.config import ModelConfig

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization
    
    RMSNorm比LayerNorm更简单，只做缩放不做平移
    公式: x * rsqrt(mean(x^2) + eps) * weight
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """计算RMS归一化"""
        # 计算平方的均值，然后取平方根的倒数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先转为float32进行数值稳定的计算，然后转回原始类型
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)
    
    为查询和键向量添加位置信息，通过旋转的方式编码相对位置
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # 预计算旋转频率
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len: int):
        """构建旋转频率缓存"""
        # 计算频率 1/theta^(2i/d)
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # 生成位置序列
        t = torch.arange(max_seq_len).float()
        
        # 计算位置与频率的外积
        freqs = torch.outer(t, freqs)
        
        # 计算cos和sin
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)                               
        
        # 注册为缓冲区（不参与梯度计算）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回对应序列长度的cos和sin值"""
        return self.freqs_cos[:seq_len], self.freqs_sin[:seq_len]
    
def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用旋转位置编码到查询和键向量
    
    Args:
        q: 查询向量 [batch, seq_len, n_heads, head_dim]
        k: 键向量 [batch, seq_len, n_kv_heads, head_dim]
        cos: 余弦值 [seq_len, head_dim]
        sin: 正弦值 [seq_len, head_dim]
    
    Returns:
        旋转后的查询和键向量
    """
    # 重新调整cos和sin的维度以匹配q和k
    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])  # [1, seq_len, 1, head_dim]
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])
    
    # 将最后一个维度分成两部分（实部和虚部）
    q_r, q_i = q.chunk(2, dim=-1)
    k_r, k_i = k.chunk(2, dim=-1)
    
    # 应用旋转变换
    q_out = torch.cat([
        q_r * cos - q_i * sin,
        q_r * sin + q_i * cos
    ], dim=-1)
    
    k_out = torch.cat([
        k_r * cos - k_i * sin,
        k_r * sin + k_i * cos
    ], dim=-1)
    
    return q_out, k_out

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复键值向量以匹配查询头数（用于Grouped Query Attention）
    
    Args:
        x: 键或值向量 [batch, seq_len, n_kv_heads, head_dim]
        n_rep: 重复次数
    
    Returns:
        重复后的向量 [batch, seq_len, n_heads, head_dim]
    """
    if n_rep == 1:
        return x
    
    batch, seq_len, n_kv_heads, head_dim = x.shape
    
    # 在第4个维度插入重复
    x = x[:, :, :, None, :].expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
    
    # 重新调整形状
    return x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)

class MLP(nn.Module):
    """多层感知机 (MLP)
    
    使用SwiGLU激活函数: SwiGLU(x) = Swish(W1*x) * (W3*x)
    其中 Swish(x) = x * sigmoid(x) = x * SiLU(x)
    """
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        
        # 三个线性层
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)  # W1
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)  # W2
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)    # W3
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: SwiGLU(x) = SiLU(W1*x) * (W3*x)"""
        gate = F.silu(self.gate_proj(x))  # SiLU激活
        up = self.up_proj(x)              # 线性变换
        out = gate * up                   # 门控机制
        return self.dropout(self.down_proj(out))
    
class MultiHeadAttention(nn.Module):
    """多头注意力机制
    
    支持：
    - 标准多头注意力 (MHA)
    - 分组查询注意力 (GQA) 
    - 多查询注意力 (MQA)
    """
    
    def __init__(self, config):
        super().__init__()
        
        # 基础配置
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = self.dim // self.n_heads
        self.dropout = config.dropout
        
        # 计算重复次数（用于GQA）
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        
        # 投影层
        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 检查是否支持Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        if not self.flash:
            print("Warning: Flash Attention不可用，使用标准实现")
            # 创建因果掩码
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool(),
                persistent=False
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cos: torch.Tensor, 
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch, seq_len, dim]
            freqs_cos: 旋转编码cos值 [seq_len, head_dim]
            freqs_sin: 旋转编码sin值 [seq_len, head_dim]
            mask: 可选的注意力掩码
        
        Returns:
            输出张量 [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 投影得到Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, n_heads * head_dim]
        k = self.k_proj(x)  # [batch, seq_len, n_kv_heads * head_dim]
        v = self.v_proj(x)  # [batch, seq_len, n_kv_heads * head_dim]
        
        # 2. 重塑为多头形式
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # 3. 应用旋转位置编码
        q, k = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)
        
        # 4. 为GQA重复K, V
        if self.n_rep > 1:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
        
        # 5. 转置为注意力计算的形式 [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 6. 计算注意力
        if self.flash:
            # 使用Flash Attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True if mask is None else False
            )
        else:
            # 手动实现注意力
            attn_output = self._manual_attention(q, k, v, mask, seq_len)
        
        # 7. 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.head_dim)
        
        # 8. 输出投影
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output
    
    def _manual_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor],
        seq_len: int
    ) -> torch.Tensor:
        """手动实现的注意力计算"""
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用因果掩码
        if mask is None:
            # 使用预设的因果掩码
            causal_mask = self.causal_mask[:seq_len, :seq_len]
            scores = scores.masked_fill(causal_mask, float('-inf'))
        else:
            # 使用自定义掩码
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output
    
class KVCache:
    """键值缓存，用于推理加速"""
    
    def __init__(self, max_batch_size: int, max_seq_len: int, n_kv_heads: int, head_dim: int, dtype=torch.float16):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        
        # 初始化缓存
        self.cache_k = torch.zeros(
            (max_batch_size, max_seq_len, n_kv_heads, head_dim),
            dtype=dtype
        )
        self.cache_v = torch.zeros(
            (max_batch_size, max_seq_len, n_kv_heads, head_dim),
            dtype=dtype
        )
        self.cache_len = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor, start_pos: int = 0):
        """更新缓存"""
        batch_size, seq_len, n_kv_heads, head_dim = k.shape
        
        # 确保缓存在正确的设备上
        if self.cache_k.device != k.device:
            self.cache_k = self.cache_k.to(k.device)
            self.cache_v = self.cache_v.to(k.device)
        
        # 更新缓存
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = k
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = v
        self.cache_len = max(self.cache_len, start_pos + seq_len)
    
    def get(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取缓存的键值"""
        return (
            self.cache_k[:batch_size, :seq_len],
            self.cache_v[:batch_size, :seq_len]
        )
    
    def reset(self):
        """重置缓存"""
        self.cache_len = 0

class CachedMultiHeadAttention(MultiHeadAttention):
    """支持KV缓存的多头注意力"""
    
    def __init__(self, config):
        super().__init__(config)
        self.kv_cache = None
    
    def setup_cache(self, max_batch_size: int, max_seq_len: int, dtype=torch.float16):
        """设置KV缓存"""
        self.kv_cache = KVCache(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            dtype=dtype
        )
    
    def forward_with_cache(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """带缓存的前向传播"""
        batch_size, seq_len, _ = x.shape
        
        # 投影得到Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # 应用旋转位置编码
        q, k = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)
        
        # 更新缓存
        if self.kv_cache is not None:
            self.kv_cache.update(k, v, start_pos)
            # 获取完整的键值序列
            k, v = self.kv_cache.get(batch_size, start_pos + seq_len)
        
        # 为GQA重复K, V
        if self.n_rep > 1:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
        
        # 转置
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力（只关注到当前位置）
        if self.flash:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=0.0,  # 推理时不使用dropout
                is_causal=True if mask is None else False
            )
        else:
            attn_output = self._manual_attention_cached(q, k, v, start_pos, seq_len)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.head_dim)
        
        # 输出投影
        output = self.o_proj(attn_output)
        
        return output
    
    def _manual_attention_cached(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        start_pos: int,
        seq_len: int
    ) -> torch.Tensor:
        """缓存模式的手动注意力计算"""
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用因果掩码 - 只能看到当前位置及之前的位置
        total_len = start_pos + seq_len
        if hasattr(self, 'causal_mask'):
            causal_mask = self.causal_mask[start_pos:total_len, :total_len]
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output    
    
class TransformerBlock(nn.Module):
    """Transformer解码器块
    
    结构：
    x -> RMSNorm -> Attention -> Add -> RMSNorm -> MLP -> Add -> output
    """
    
    def __init__(self, layer_id: int, config):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.dim
        
        # 注意力机制
        self.attention = MultiHeadAttention(config)
        
        # MLP
        self.mlp = MLP(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        
        # 归一化层
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch, seq_len, dim]
            freqs_cos: 旋转编码cos值 [seq_len, head_dim]
            freqs_sin: 旋转编码sin值 [seq_len, head_dim]
            mask: 可选的注意力掩码
        
        Returns:
            输出张量 [batch, seq_len, dim]
        """
        # 1. 自注意力子层 (Pre-Norm)
        # x + Attention(RMSNorm(x))
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos,
            freqs_sin,
            mask
        )
        
        # 2. 前馈网络子层 (Pre-Norm)
        # h + MLP(RMSNorm(h))
        out = h + self.mlp(self.ffn_norm(h))
        
        return out
    
class CachedTransformerBlock(nn.Module):
    """支持KV缓存的Transformer解码器块"""
    
    def __init__(self, layer_id: int, config):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.dim
        
        # 缓存注意力机制
        self.attention = CachedMultiHeadAttention(config)
        
        # MLP
        self.mlp = MLP(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        
        # 归一化层
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
    
    def setup_cache(self, max_batch_size: int, max_seq_len: int, dtype=torch.float16):
        """设置KV缓存"""
        self.attention.setup_cache(max_batch_size, max_seq_len, dtype)
    
    def forward_with_cache(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """带缓存的前向传播"""
        # 自注意力子层
        h = x + self.attention.forward_with_cache(
            self.attention_norm(x),
            freqs_cos,
            freqs_sin,
            start_pos,
            mask
        )
        
        # 前馈网络子层
        out = h + self.mlp(self.ffn_norm(h))
        
        return out
    
class Transformer(nn.Module):
    """完整的Transformer语言模型
    
    支持：
    - 训练模式：完整的序列到序列处理
    - 推理模式：增量生成，支持KV缓存
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.dim = config.dim
        self.max_seq_len = config.max_seq_len
        
        # === 核心组件 ===
        # 1. 词嵌入层
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # 2. Dropout层
        self.dropout = nn.Dropout(config.dropout)
        
        # 3. 旋转位置编码
        self.rope = RotaryEmbedding(
            dim=config.dim // config.n_heads,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta
        )
        
        # 4. Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(layer_id, config)
            for layer_id in range(config.n_layers)
        ])
        
        # 5. 最终归一化层
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # 6. 输出投影层
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # === 权重共享 ===
        # 输入嵌入和输出投影共享权重
        self.tok_embeddings.weight = self.output.weight
        
        # === 初始化 ===
        self.apply(self._init_weights)
        
        # 对特定层应用缩放初始化
        for name, param in self.named_parameters():
            if name.endswith('o_proj.weight') or name.endswith('down_proj.weight'):
                # 对输出投影层应用缩放，有助于训练稳定性
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))
        
        # 训练状态
        self.last_loss = None
        
        # 推理状态
        self.inference_mode = False
        self.cached_layers = None
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            # 线性层使用正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 嵌入层使用正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def setup_inference_cache(self, max_batch_size: int, max_seq_len: int, dtype=torch.float16):
        """设置推理缓存"""
        self.cached_layers = nn.ModuleList([
            CachedTransformerBlock(layer_id, self.config)
            for layer_id in range(self.config.n_layers)
        ])
        
        # 复制训练权重到缓存层
        for cached_layer, train_layer in zip(self.cached_layers, self.layers):
            cached_layer.load_state_dict(train_layer.state_dict())
            cached_layer.setup_cache(max_batch_size, max_seq_len, dtype)
        
        self.inference_mode = True
        print(f"推理缓存设置完成: max_batch_size={max_batch_size}, max_seq_len={max_seq_len}")
    
    def forward(
        self, 
        tokens: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """训练模式的前向传播
        
        Args:
            tokens: 输入token [batch, seq_len]
            targets: 目标token [batch, seq_len]，用于计算损失
            mask: 可选的注意力掩码
        
        Returns:
            logits: [batch, seq_len, vocab_size] 或 [batch, 1, vocab_size]
        """
        batch_size, seq_len = tokens.shape
        
        # 1. 词嵌入
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        
        # 2. 获取旋转位置编码
        freqs_cos, freqs_sin = self.rope(h, seq_len)
        
        # 3. 通过Transformer层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, mask)
        
        # 4. 最终归一化
        h = self.norm(h)
        
        # 5. 输出投影
        if targets is not None:
            # 训练模式：计算所有位置的logits
            logits = self.output(h)
            # 计算损失
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
        else:
            # 推理模式：只计算最后一个位置的logits
            logits = self.output(h[:, [-1], :])
            self.last_loss = None
        
        return logits
    
    def forward_with_cache(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0
    ) -> torch.Tensor:
        """使用KV缓存的前向传播（推理专用）"""
        if not self.inference_mode or self.cached_layers is None:
            raise RuntimeError("请先调用setup_inference_cache()设置推理缓存")
        
        batch_size, seq_len = tokens.shape
        
        # 1. 词嵌入
        h = self.tok_embeddings(tokens)
        
        # 2. 获取旋转位置编码（从start_pos开始）
        total_len = start_pos + seq_len
        freqs_cos, freqs_sin = self.rope(h, total_len)
        
        # 只取当前序列对应的编码
        freqs_cos = freqs_cos[start_pos:total_len]
        freqs_sin = freqs_sin[start_pos:total_len]
        
        # 3. 通过缓存的Transformer层
        for layer in self.cached_layers:
            h = layer.forward_with_cache(h, freqs_cos, freqs_sin, start_pos)
        
        # 4. 最终归一化和输出
        h = self.norm(h)
        logits = self.output(h)
        
        return logits
    
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        use_cache: bool = True
    ) -> torch.Tensor:
        """文本生成
        
        Args:
            prompt_tokens: 提示token [batch, prompt_len]
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            top_k: top-k采样
            top_p: nucleus采样
            do_sample: 是否采样，False则使用贪心解码
            use_cache: 是否使用KV缓存
        
        Returns:
            generated_tokens: [batch, prompt_len + max_new_tokens]
        """
        batch_size, prompt_len = prompt_tokens.shape
        
        # 检查序列长度
        if prompt_len + max_new_tokens > self.max_seq_len:
            print(f"Warning: 总长度({prompt_len + max_new_tokens})超过最大长度({self.max_seq_len})")
        
        # 初始化生成序列
        generated = prompt_tokens.clone()
        
        if use_cache and self.inference_mode:
            # 使用缓存的推理
            return self._generate_with_cache(
                prompt_tokens, max_new_tokens, temperature, top_k, top_p, do_sample
            )
        else:
            # 不使用缓存的推理
            return self._generate_without_cache(
                prompt_tokens, max_new_tokens, temperature, top_k, top_p, do_sample
            )
    
    def _generate_with_cache(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool
    ) -> torch.Tensor:
        """使用KV缓存的生成"""
        batch_size, prompt_len = prompt_tokens.shape
        generated = prompt_tokens.clone()
        
        # 1. 处理提示序列
        logits = self.forward_with_cache(prompt_tokens, start_pos=0)
        next_token = self._sample_next_token(
            logits[:, -1, :], temperature, top_k, top_p, do_sample
        )
        generated = torch.cat([generated, next_token], dim=1)
        
        # 2. 逐个生成新token
        for i in range(max_new_tokens - 1):
            logits = self.forward_with_cache(next_token, start_pos=prompt_len + i)
            next_token = self._sample_next_token(
                logits[:, -1, :], temperature, top_k, top_p, do_sample
            )
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def _generate_without_cache(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool
    ) -> torch.Tensor:
        """不使用KV缓存的生成"""
        generated = prompt_tokens.clone()
        
        for _ in range(max_new_tokens):
            # 前向传播整个序列
            logits = self.forward(generated)
            
            # 采样下一个token
            next_token = self._sample_next_token(
                logits[:, -1, :], temperature, top_k, top_p, do_sample
            )
            
            # 添加到生成序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查长度限制
            if generated.shape[1] >= self.max_seq_len:
                break
        
        return generated
    
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool
    ) -> torch.Tensor:
        """采样下一个token"""
        if not do_sample:
            # 贪心解码
            return torch.argmax(logits, dim=-1, keepdim=True)
        
        # 应用温度
        if temperature > 0:
            logits = logits / temperature
        
        # Top-k 采样
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            # 找到top-k的最小值，将其他值设为-inf
            top_k_values, _ = torch.topk(logits, top_k, dim=-1)
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_top_k, 
                               torch.tensor(float('-inf'), device=logits.device),
                               logits)
        
        # Top-p (nucleus) 采样
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 找到累积概率超过top_p的位置
            sorted_indices_to_remove = cumulative_probs > top_p
            # 保留第一个超过top_p的token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            # 在原始logits上应用掩码
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def get_num_params(self, non_embedding=True):
        """获取参数数量"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """估计模型浮点运算利用率 (MFU)"""
        # 首先估计每次前向传播的浮点运算数
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # 表达为每秒浮点运算数
        flops_achieved = flops_per_iter * (1.0/dt)
        # 硬件的峰值浮点运算数，对于A100约为312 TFLOPS
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """从预训练模型加载"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 从checkpoint中恢复配置
        config = ModelConfig(**checkpoint['config'])
        
        # 创建模型
        model = cls(config)
        
        # 加载权重
        model.load_state_dict(checkpoint['model'])
        
        return model
    
    def save_checkpoint(self, path: str, optimizer=None, iter_num=None, best_val_loss=None):
        """保存检查点"""
        checkpoint = {
            'model': self.state_dict(),
            'config': self.config.__dict__,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
        }
        
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"检查点已保存到: {path}")
