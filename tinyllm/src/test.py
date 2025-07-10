# inference.py - 修复版本
import torch
import torch.nn.functional as F
from model import Transformer
from data_utils import TinyStoriesTokenizer
from config.config import ModelConfig
import os
import argparse
import time

class TextGenerator:
    def __init__(self, checkpoint_path, vocab_path, device='cuda:4'):
        """
        初始化文本生成器
        Args:
            checkpoint_path: 模型检查点路径
            vocab_path: 词汇表路径
            device: 设备
        """
        self.device = device
        
        # 加载tokenizer
        print("加载tokenizer...")
        self.tokenizer = TinyStoriesTokenizer(vocab_path)
        print(f"✅ tokenizer加载成功，词汇表大小: {self.tokenizer.vocab_size}")
        
        # 加载模型
        print("加载模型...")
        self.model, self.model_config = self._load_model(checkpoint_path)
        print("✅ 模型加载成功")
        
        # 设置为评估模式
        self.model.eval()
    
    def _load_model(self, checkpoint_path):
        """加载模型"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        print(f"从检查点加载模型: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 打印检查点信息
        print(f"检查点键: {list(checkpoint.keys())}")
        if 'iter_num' in checkpoint:
            print(f"训练步数: {checkpoint['iter_num']}")
        if 'best_val_loss' in checkpoint:
            print(f"最佳验证损失: {checkpoint['best_val_loss']}")
        
        # 获取模型配置
        if 'model_config' in checkpoint:
            model_config_dict = checkpoint['model_config']
            model_config = ModelConfig(**model_config_dict)
            print("✅ 从检查点加载模型配置")
        else:
            # 如果没有配置，使用默认配置
            print("⚠️  检查点中没有找到模型配置，使用默认配置")
            model_config = ModelConfig(
                dim=288,
                n_layers=6,
                n_heads=6,
                n_kv_heads=6,
                hidden_dim=1024,
                vocab_size=self.tokenizer.vocab_size,
                max_seq_len=256
            )
        
        # 确保词汇表大小匹配
        model_config.vocab_size = self.tokenizer.vocab_size
        
        print(f"模型配置: {model_config.__dict__}")
        
        # 创建模型
        model = Transformer(model_config).to(self.device)
        
        # 处理模型权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 检查是否需要处理torch.compile的前缀
        sample_key = list(state_dict.keys())[0]
        if sample_key.startswith('_orig_mod.'):
            print("🔧 检测到torch.compile前缀，正在处理...")
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key[10:]  # 移除'_orig_mod.'前缀
                new_state_dict[new_key] = value
            state_dict = new_state_dict
            print("✅ 前缀处理完成")
        
        # 验证权重键
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            print(f"⚠️  缺少的权重键: {list(missing_keys)[:5]}...")  # 只显示前5个
        if unexpected_keys:
            print(f"⚠️  意外的权重键: {list(unexpected_keys)[:5]}...")  # 只显示前5个
        
        # 加载权重
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ 模型权重加载成功")
        except Exception as e:
            print(f"❌ 严格模式加载失败: {e}")
            try:
                model.load_state_dict(state_dict, strict=False)
                print("⚠️  非严格模式加载成功，可能存在部分权重不匹配")
            except Exception as e2:
                print(f"❌ 模型权重加载完全失败: {e2}")
                raise e2
        
        print(f"模型参数量: {model.get_num_params():,}")
        
        return model, model_config
    
    def generate(self, prompt, max_length=100, temperature=0.8, top_k=40, top_p=0.9, 
                 do_sample=True, repetition_penalty=1.1):
        """生成文本"""
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # 记录原始长度
        original_length = input_ids.size(1)
        
        print(f"输入prompt: '{prompt}'")
        print(f"输入tokens: {input_ids.tolist()[0]}")
        print(f"输入长度: {original_length}")
        print("开始生成...")
        
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(max_length):
                # 检查序列长度 - 修复这里
                if input_ids.size(1) >= self.model_config.max_seq_len:
                    print(f"达到最大序列长度({self.model_config.max_seq_len})，停止生成")
                    break
                
                try:
                    # 获取logits
                    logits = self.model(input_ids)
                    
                    # 只取最后一个位置的logits
                    logits = logits[:, -1, :] / temperature
                    
                    # 应用重复惩罚
                    if repetition_penalty != 1.0:
                        logits = self._apply_repetition_penalty(logits, input_ids, repetition_penalty)
                    
                    # 采样策略
                    if do_sample:
                        # Top-k采样
                        if top_k > 0:
                            logits = self._top_k_filtering(logits, top_k)
                        
                        # Top-p采样
                        if top_p < 1.0:
                            logits = self._top_p_filtering(logits, top_p)
                        
                        # 随机采样
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        # 贪心采样
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    next_token_id = next_token.item()
                    
                    # 检查是否生成EOS
                    if next_token_id == self.tokenizer.eos_id:
                        print("生成了EOS token，停止生成")
                        break
                    
                    # 记录生成的token
                    generated_tokens.append(next_token_id)
                    
                    # 添加到序列
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    # 打印进度
                    if step % 10 == 0 or step < 5:
                        try:
                            current_text = self.tokenizer.decode(generated_tokens)
                            print(f"Step {step}: {current_text}")
                        except:
                            print(f"Step {step}: [解码失败, token: {next_token_id}]")
                
                except Exception as e:
                    print(f"生成step {step}时出错: {e}")
                    break
        
        # 解码生成的部分
        try:
            generated_text = self.tokenizer.decode(generated_tokens)
            full_text = prompt + generated_text
        except Exception as e:
            print(f"解码失败: {e}")
            generated_text = "[解码失败]"
            full_text = prompt + generated_text
        
        print(f"\n生成完成，生成了 {len(generated_tokens)} 个tokens")
        print(f"生成的tokens: {generated_tokens}")
        
        return {
            'generated_text': generated_text,
            'full_text': full_text,
            'generated_tokens': generated_tokens,
            'input_tokens': input_ids[0, :original_length].tolist(),
            'total_tokens': len(generated_tokens)
        }
    
    def _apply_repetition_penalty(self, logits, input_ids, penalty):
        """应用重复惩罚"""
        for token_id in set(input_ids[0].tolist()):
            logits[0, token_id] /= penalty
        return logits
    
    def _top_k_filtering(self, logits, top_k):
        """Top-k过滤"""
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1:].expand_as(logits)
        return torch.where(logits < min_values, 
                          torch.full_like(logits, float('-inf')), 
                          logits)
    
    def _top_p_filtering(self, logits, top_p):
        """Top-p过滤"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除超过top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 创建mask
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits

def main():
    parser = argparse.ArgumentParser(description='TinyLLM推理测试')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_checkpoint.ckpt',
                        help='模型检查点路径')
    parser.add_argument('--vocab', type=str, default='/home/suxin/tinyllm/src/TinyStories4096.model',
                        help='词汇表路径')
    parser.add_argument('--device', type=str, default='cuda:4',
                        help='设备')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='生成提示')
    parser.add_argument('--max_length', type=int, default=50,
                        help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='温度参数')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k采样')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p采样')
    
    args = parser.parse_args()
    
    try:
        # 创建生成器
        generator = TextGenerator(args.checkpoint, args.vocab, args.device)
        
        # 单次生成模式
        print("🚀 开始生成文本")
        print("=" * 50)
        
        result = generator.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        print(f"\n📝 生成结果:")
        print(f"输入: {args.prompt}")
        print(f"生成: {result['generated_text']}")
        print(f"完整: {result['full_text']}")
        print(f"生成tokens数: {result['total_tokens']}")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
