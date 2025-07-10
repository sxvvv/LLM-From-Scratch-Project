import os
import pyarrow.parquet as pq
from pathlib import Path
import sentencepiece as spm
import tempfile

def train_vocab(data_dir, vocab_size=4096, model_name="vocab"):
    """
    Args:
        data_dir:数据目录路径
        vocab_size:词汇表大小
        model_name:模型名
    Returns:
        模型文件路径
    """
    print(f"开始训练词汇表，词汇表大小： {vocab_size}")

    print("Step 1: 准备训练语料...")
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
    data_path = Path(data_dir)
    parquet_files = list(data_path.glob("train-*.parquet"))
    
    text_count = 0
    for file_path in parquet_files:
        print(f"处理文件: {file_path.name}")
        
        # 读取parquet文件
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(columns=["text"], batch_size=50000):
            texts = batch["text"].to_pylist()
            
            for text in texts:
                if text and len(str(text).strip()) > 10:  # 过滤太短的文本
                    temp_file.write(str(text).strip() + '\n')
                    text_count += 1
                    
                    if text_count % 10000 == 0:
                        print(f"已处理 {text_count} 条文本")
    
    temp_file.close()
    print(f"语料准备完成，共 {text_count} 条文本")
    
    # 步骤2: 训练SentencePiece模型
    print("Step 2: 训练SentencePiece模型...")
    
    try:
        spm.SentencePieceTrainer.train(
            input=temp_file.name,
            model_prefix=model_name,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            num_threads=os.cpu_count(),
            pad_id=0,
            unk_id=1, 
            bos_id=2,
            eos_id=3,
            input_sentence_size=10_000_000,
            shuffle_input_sentence=True
        )
        
        print(f"✅ 词汇表训练完成!")
        print(f"模型文件: {model_name}.model")
        print(f"词汇文件: {model_name}.vocab")
        
    finally:
        # 清理临时文件
        os.unlink(temp_file.name)
    
    return f"{model_name}.model"

model_path = train_vocab("/home/suxin/tinyllm/data/TinyStories/data", vocab_size=4096, model_name="TinyStories4096")