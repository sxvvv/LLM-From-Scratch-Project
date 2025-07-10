import os
import time
import math
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import swanlab
from config.config import ModelConfig
from model import Transformer
from data_utils import create_data_loader, load_pretrained_model


class TrainerConfig:
    """训练配置"""
    # 数据
    dataset = 'tinystories'
    batch_size = 32
    max_seq_len = 256
    
    # 训练
    max_iters = 50000
    eval_interval = 500
    eval_iters = 100
    log_interval = 100
    
    # 优化器
    learning_rate = 5e-4
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    
    # 学习率调度
    decay_lr = True
    warmup_iters = 1000
    lr_decay_iters = 5000
    min_lr = 5e-5
    
    # 系统
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    compile = True
    
    # 保存
    always_save_checkpoint = False
    
    # 分布式训练
    backend = 'nccl'
    
    # 文件路径
    data_dir = '/home/suxin/tinyllm/data/TinyStories/data'
    vocab_path = '/home/suxin/tinyllm/src/TinyStories4096.model'
    pretrained_model = '/home/suxin/tinyllm/src/TinyStories4096.model'

    # 监控配置
    swanlab_project = "TinyStories"
    log_interval = 10
    out_dir = "checkpoints"
    use_swanlab: bool = False

class Trainer:
    """训练器"""
    
    def __init__(self, model_config: ModelConfig, train_config: TrainerConfig):
        self.model_config = model_config
        self.train_config = train_config
        
        # 设置设备
        self.device = train_config.device
        self.dtype = train_config.dtype
        
        # 分布式训练设置
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend=train_config.backend)
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
        else:
            self.master_process = True
            self.ddp_world_size = 1
        
        # 设置随机种子
        torch.manual_seed(1337)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 创建输出目录
        if self.master_process:
            os.makedirs(train_config.out_dir, exist_ok=True)
        
        # 初始化模型
        self.model = Transformer(model_config).to(self.device)
        
        # 尝试加载预训练权重
        if os.path.exists(train_config.pretrained_model):
            pretrained_weights = load_pretrained_model(train_config.pretrained_model)
            if pretrained_weights is not None:
                try:
                    self.model.load_state_dict(pretrained_weights, strict=False)
                    print("成功加载预训练权重")
                except Exception as e:
                    print(f"加载预训练权重失败: {e}")
        
        # 编译模型
        if train_config.compile:
            print("编译模型...")
            self.model = torch.compile(self.model)
        
        # DDP包装
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        
        # 获取原始模型引用
        self.raw_model = self.model.module if self.ddp else self.model
        
        # 初始化优化器
        self.optimizer = self._configure_optimizer()
        # 初始化SwanLab（可选）
        if train_config.use_swanlab:
            try:
                swanlab.init(
                    project=train_config.swanlab_project,
                    experiment_name=f"tinyllm_{train_config.out_dir}",
                    config={
                        "model_config": model_config.__dict__,
                        "train_config": train_config.__dict__
                    }
                )
                self.use_swanlab = True
                print("SwanLab初始化成功")
            except Exception as e:
                print(f"SwanLab初始化失败: {e}")
                self.use_swanlab = False
        else:
            self.use_swanlab = False
        
        print(f"模型参数量: {self.raw_model.get_num_params():,}")

    
    def _configure_optimizer(self):
        """配置优化器"""
        # 分离需要权重衰减和不需要权重衰减的参数
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and 'norm' in mn.lower():  # 任何包含'norm'的模块的weight都不衰减
                    no_decay.add(fpn)
        
        # 获取所有参数
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        
        # 验证所有参数都被分类
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"参数不能同时在decay和no_decay中: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, f"参数未被分类: {param_dict.keys() - union_params}"
        
        # 创建优化器组 - 直接使用原始参数名
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict], "weight_decay": self.train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict], "weight_decay": 0.0},
        ]
        
        # 使用AdamW
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.train_config.learning_rate,
            betas=(self.train_config.beta1, self.train_config.beta2)
        )
        
        return optimizer


    
    def get_lr(self, iter_num):
        """获取学习率（带预热和衰减）"""
        # 预热阶段
        if iter_num < self.train_config.warmup_iters:
            return self.train_config.learning_rate * iter_num / self.train_config.warmup_iters
        
        # 衰减阶段
        if iter_num > self.train_config.lr_decay_iters:
            return self.train_config.min_lr
        
        # 余弦衰减
        decay_ratio = (iter_num - self.train_config.warmup_iters) / (self.train_config.lr_decay_iters - self.train_config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.train_config.min_lr + coeff * (self.train_config.learning_rate - self.train_config.min_lr)
    
    def train(self, get_batch_fn):
        """训练主循环"""
        iter_num = 0
        best_val_loss = float('inf')
        
        print("开始训练...")
        print(f"最大迭代次数: {self.train_config.max_iters}")
        print(f"批次大小: {self.train_config.batch_size}")
        print(f"序列长度: {self.train_config.max_seq_len}")
        
        while iter_num < self.train_config.max_iters:
            # 设置学习率
            lr = self.get_lr(iter_num) if self.train_config.decay_lr else self.train_config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # 评估
            if iter_num % self.train_config.eval_interval == 0 and self.master_process:
                val_loss = self.evaluate(get_batch_fn)
                print(f"步骤 {iter_num}: 验证损失 {val_loss:.4f}, 学习率 {lr:.6f}")
                
                # 添加swanlab验证损失记录（在这里加）
                if hasattr(self, 'use_swanlab') and self.use_swanlab:
                    swanlab.log({
                        "val/loss": val_loss,
                        "val/best_loss": best_val_loss,
                        "train/step": iter_num
                    })
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if iter_num > 0:
                        self.save_checkpoint(iter_num, best_val_loss)
                        print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")

            
            # 训练步骤
            t0 = time.time()
            
            # 前向传播
            try:
                X, Y = get_batch_fn('train', self.train_config.batch_size)
                X, Y = X.to(self.device), Y.to(self.device)
                
                # 使用混合精度
                with torch.amp.autocast(device_type=self.device.split(':')[0], dtype=self.dtype):
                    logits = self.model(X, Y)
                    loss = self.raw_model.last_loss
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 检测到无效损失 {loss}, 跳过此步骤")
                    iter_num += 1
                    continue
                
                # 反向传播
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                # 梯度裁剪
                if self.train_config.grad_clip != 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
                
                # 优化器步骤
                self.optimizer.step()
                
                # 记录
                t1 = time.time()
                dt = t1 - t0
                
                if iter_num % self.train_config.log_interval == 0 and self.master_process:
                    lossf = loss.item()
                    if hasattr(self.raw_model, 'estimate_mfu'):
                        mfu = self.raw_model.estimate_mfu(self.train_config.batch_size * self.ddp_world_size, dt)
                        print(f"步骤 {iter_num}: 损失 {lossf:.4f}, 时间 {dt*1000:.2f}ms, MFU {mfu*100:.2f}%")
                        # 添加swanlab日志记录（在这里加）
                        if hasattr(self, 'use_swanlab') and self.use_swanlab:
                            swanlab.log({
                                "train/loss": lossf,
                                "train/learning_rate": lr,
                                "train/time_per_step": dt * 1000,
                                "train/mfu": mfu * 100,
                                "train/step": iter_num
                            })
                    else:
                        print(f"步骤 {iter_num}: 损失 {lossf:.4f}, 时间 {dt*1000:.2f}ms")
                        # 添加swanlab日志记录（在这里加）
                        if hasattr(self, 'use_swanlab') and self.use_swanlab:
                            swanlab.log({
                                "train/loss": lossf,
                                "train/learning_rate": lr,
                                "train/time_per_step": dt * 1000,
                                "train/step": iter_num
                            })
                
            except Exception as e:
                print(f"训练步骤出错: {e}")
                iter_num += 1
                continue
            
            iter_num += 1
        
        # 训练结束，保存最终模型
        if self.master_process:
            self.save_checkpoint(iter_num, best_val_loss, final=True)
            print("训练完成，保存最终模型")
            if hasattr(self, 'use_swanlab') and self.use_swanlab:
                swanlab.finish()


        if self.ddp:
            destroy_process_group()
    
    @torch.no_grad()
    def evaluate(self, get_batch_fn):
        """评估模型"""
        self.model.eval()
        losses = torch.zeros(self.train_config.eval_iters)
        
        for k in range(self.train_config.eval_iters):
            try:
                X, Y = get_batch_fn('val', self.train_config.batch_size)
                X, Y = X.to(self.device), Y.to(self.device)
                
                with torch.amp.autocast(device_type=self.device.split(':')[0], dtype=self.dtype):
                    logits = self.model(X, Y)
                    loss = self.raw_model.last_loss
                
                losses[k] = loss.item()
                
            except Exception as e:
                print(f"评估步骤出错: {e}")
                losses[k] = float('inf')
        
        self.model.train()
        return losses.mean()
    
    def save_checkpoint(self, iter_num, best_val_loss, final=False):
        """保存检查点"""
        checkpoint_name = 'final_checkpoint.ckpt' if final else 'checkpoint.ckpt'
        checkpoint_path = os.path.join(self.train_config.out_dir, checkpoint_name)
        
        # 创建检查点字典
        checkpoint = {
            'model_state_dict': self.raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model_config.__dict__,
            'train_config': self.train_config.__dict__,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存到: {checkpoint_path}")


def main():
    """主函数"""
    # 配置
    model_config = ModelConfig(
        dim=288,
        n_layers=6,
        n_heads=6,
        n_kv_heads=6,  
        hidden_dim=1024,
        vocab_size=4096,
        max_seq_len=256
    )
    
    train_config = TrainerConfig()
    train_config.use_swanlab = True 
    # 创建数据加载器
    try:
        print("正在创建数据加载器...")
        get_batch_fn, tokenizer = create_data_loader(
            train_config.data_dir,
            train_config.vocab_path,
            train_config.max_seq_len
        )
        print("数据加载器创建成功")
        
        # 更新词汇表大小
        model_config.vocab_size = tokenizer.vocab_size
        print(f"更新词汇表大小为: {model_config.vocab_size}")
        
        # 创建训练器并开始训练
        trainer = Trainer(model_config, train_config)
        trainer.train(get_batch_fn)
        
    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()
        
        # swanlab关闭
        try:
            if 'trainer' in locals() and hasattr(trainer, 'use_swanlab') and trainer.use_swanlab:
                swanlab.finish()
        except:
            pass

if __name__ == '__main__':
    main()
