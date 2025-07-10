# data_utils.py
import os
import torch
import sentencepiece as spm
import pyarrow.parquet as pq
from pathlib import Path
import random

class TinyStoriesTokenizer:
    def __init__(self, model_path):
        """
        初始化tokenizer
        Args:
            model_path: SentencePiece模型文件路径(.model文件)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        if not model_path.endswith('.model'):
            raise ValueError(f"请提供.model文件路径，而不是: {model_path}")
        
        self.sp = spm.SentencePieceProcessor()
        
        try:
            self.sp.load(model_path)
            print(f"✅ 成功加载SentencePiece模型: {model_path}")
        except Exception as e:
            raise RuntimeError(f"加载SentencePiece模型失败: {e}")
        
        # 获取特殊token ID
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        
        print(f"词汇表大小: {self.sp.vocab_size()}")
        print(f"特殊token - PAD: {self.pad_id}, UNK: {self.unk_id}, BOS: {self.bos_id}, EOS: {self.eos_id}")
    
    @property
    def vocab_size(self):
        return self.sp.vocab_size()
    
    def encode(self, text, add_bos=True, add_eos=True):
        """编码文本"""
        if not isinstance(text, str):
            text = str(text)
        
        # 基础编码
        tokens = self.sp.encode(text)
        
        # 添加特殊token
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
            
        return tokens
    
    def decode(self, tokens):
        """解码token"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # 移除特殊token（保留内容中的特殊token）
        filtered_tokens = []
        for token in tokens:
            if token == self.eos_id:  # 遇到EOS就停止
                break
            if token not in [self.pad_id, self.bos_id]:
                filtered_tokens.append(token)
        
        return self.sp.decode(filtered_tokens)
    
    def encode_batch(self, texts, add_bos=True, add_eos=True):
        """批量编码"""
        return [self.encode(text, add_bos, add_eos) for text in texts]

def create_data_loader(data_dir, vocab_path, max_seq_len):
    """创建数据加载器"""
    print(f"正在创建数据加载器...")
    print(f"数据目录: {data_dir}")
    print(f"词汇表路径: {vocab_path}")
    print(f"最大序列长度: {max_seq_len}")
    
    # 初始化tokenizer
    tokenizer = TinyStoriesTokenizer(vocab_path)
    
    # 加载并预处理数据
    print("加载训练数据...")
    train_data = load_and_tokenize_data(data_dir, tokenizer, max_seq_len, split='train')
    
    print("加载验证数据...")
    val_data = load_and_tokenize_data(data_dir, tokenizer, max_seq_len, split='val')
    
    def get_batch(split, batch_size):
        data = train_data if split == 'train' else val_data
        
        if len(data) < max_seq_len + 1:
            raise ValueError(f"数据太少，无法创建批次。数据长度: {len(data)}, 需要: {max_seq_len + 1}")
        
        # 随机选择batch_size个起始位置
        max_start_idx = len(data) - max_seq_len - 1
        indices = torch.randint(0, max_start_idx, (batch_size,))
        
        # 创建输入和标签
        x = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        y = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        
        for i, idx in enumerate(indices):
            x[i] = torch.tensor(data[idx:idx + max_seq_len], dtype=torch.long)
            y[i] = torch.tensor(data[idx + 1:idx + max_seq_len + 1], dtype=torch.long)
        
        return x, y
    
    return get_batch, tokenizer

def load_and_tokenize_data(data_dir, tokenizer, max_seq_len, split='train', max_texts=50000):
    """加载并分词数据"""
    print(f"正在加载 {split} 数据...")
    
    data_path = Path(data_dir)
    
    # 查找parquet文件
    train_files = list(data_path.glob("train-*.parquet"))
    val_files = list(data_path.glob("validation-*.parquet"))
    
    print(f"找到 {len(train_files)} 个训练文件，{len(val_files)} 个验证文件")
    
    if split == 'train':
        parquet_files = train_files
    else:
        parquet_files = val_files if val_files else train_files[-1:]  # 如果没有验证文件，使用最后一个训练文件
    
    if not parquet_files:
        raise FileNotFoundError(f"在 {data_dir} 中找不到parquet文件")
    
    all_tokens = []
    text_count = 0
    
    for file_path in parquet_files:
        print(f"处理文件: {file_path.name}")
        
        try:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(columns=["text"], batch_size=5000):
                texts = batch["text"].to_pylist()
                
                for text in texts:
                    if text and len(str(text).strip()) > 10:  # 过滤太短的文本
                        try:
                            tokens = tokenizer.encode(str(text).strip())
                            all_tokens.extend(tokens)
                            text_count += 1
                            
                            if text_count % 1000 == 0:
                                print(f"已处理 {text_count} 条文本，当前token数: {len(all_tokens)}")
                            
                            if text_count >= max_texts:
                                break
                        except Exception as e:
                            print(f"编码文本失败: {e}")
                            continue
                
                if text_count >= max_texts:
                    break
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
        
        if text_count >= max_texts:
            break
    
    if not all_tokens:
        raise ValueError("没有成功加载任何数据")
    
    print(f"✅ {split} 数据加载完成：{text_count} 条文本，{len(all_tokens)} 个token")
    
    # 如果是验证集且使用的是训练文件，取后20%作为验证
    if split == 'val' and not val_files:
        split_idx = int(len(all_tokens) * 0.8)
        all_tokens = all_tokens[split_idx:]
        print(f"使用训练数据的后20%作为验证集，验证集大小: {len(all_tokens)} tokens")
    
    return all_tokens

def load_pretrained_model(model_path):
    """加载预训练模型权重"""
    if not model_path or not os.path.exists(model_path):
        print("未找到预训练模型文件")
        return None
    
    try:
        # 检查是否是PyTorch检查点文件
        if model_path.endswith('.model'):
            print("检测到SentencePiece模型文件，跳过权重加载")
            return None
        
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            print("从检查点加载模型权重")
            return checkpoint['model_state_dict']
        else:
            print("直接加载模型权重")
            return checkpoint
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
        return None
