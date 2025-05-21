# 导入所需的库
import torch # PyTorch 深度学习库，用于构建和运行神经网络
import os # 用于操作系统相关操作，如检查文件/目录是否存在、创建目录
import logging # 用于记录日志信息，方便调试和追踪
import pandas as pd # 数据处理库，虽然在这个脚本中直接使用了列表，但 pandas 是常见的数据处理工具
import numpy as np # 数值计算库，这里用于处理模型的输出 logits 和标签

# 导入 Hugging Face 相关的库
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer # transformers 库的核心类，用于加载 Tokenizer、模型、训练参数和 Trainer
from datasets import Dataset # 从 datasets 库导入 Dataset 类，用于方便地处理和预处理数据
# 从 evaluate 库导入 load 函数来加载评估指标，如准确率和 F1 分数
from evaluate import load

# 设置日志级别和格式
# 配置日志的基本格式和级别，日志信息会打印到控制台
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 导入您的本地数据文件
# 使用 try...except 块来尝试安全地导入数据文件
try:

    data_dir = r'D:\biyesheji\ImageRecognition\trainModel' # <-- 确保这是你数据文件所在的正确路径
    import sys # 导入 sys 模块，用于修改 Python 路径
    # 将数据目录添加到 sys.path 中，这样 Python 解释器才能找到该目录下的模块
    if data_dir not in sys.path:
        sys.path.append(data_dir)


    from train_data import train_data
    from valid_data import valid_data
    from test_data import test_data

    logging.info("成功导入训练、验证、测试数据集列表。")
except ImportError as e:
    # 如果导入失败 (例如文件或变量不存在)，记录 ERROR 级别日志，并提供帮助信息
    logging.error(f"无法导入数据集文件: {e}")
    logging.error(f"请确保 {data_dir} 目录下存在 train_data.py, valid_data.py, test_data.py 文件，")
    logging.error(f"并且每个文件中包含同名的 Python 列表变量 (例如 train_data = [...])。")
    logging.error("如果您在其他目录运行此脚本，请检查 data_dir 路径是否正确并添加到 sys.path。")
    # 如果数据无法加载，训练无法进行，因此直接退出脚本
    exit()

# --- 配置 ---
# 定义预训练模型或之前训练好的检查点路径
# Tokenizer 和模型的配置 (如词表、模型结构参数) 将从这个路径加载
MODEL_PATH = r"D:\biyesheji\Biyesheji2.0\roberta\roberta-wwm-ext" # <-- 确保这是你的模型文件所在的正确路径

# 检查模型路径是否存在且是一个有效的目录
if not os.path.isdir(MODEL_PATH):
     if not os.path.isdir(os.path.abspath(MODEL_PATH)):
         logging.error(f"模型路径不存在或不是一个有效的目录：{MODEL_PATH}")
         exit()

# 检测当前系统是否可以使用 CUDA (GPU)，设置训练使用的设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 打印使用的设备信息
if DEVICE.type == 'cuda':
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("未检测到 GPU，将使用 CPU 进行训练。CPU 训练速度非常慢！")


# --- 敏感信息类别定义 (与您的 FastAPI 代码中的分类保持一致) ---
# 这个列表必须包含训练数据中的所有可能标签，包括负样本（无敏感信息）
# 同时需要与你的API推理代码中的 SENSITIVE_CATEGORIES 定义一致
PREDICTION_CATEGORIES = [
    "个人隐私信息",
    "黄色信息",
    "不良有害信息", # 合并暴力、仇恨、违法活动指引
    "无敏感信息"     # 训练时必须包含负样本，用于区分非敏感内容
]

# 创建类别名称到整数 ID 的映射字典 (用于将字符串标签转换为模型能处理的数字)
ID_TO_CATEGORY = {i: cat for i, cat in enumerate(PREDICTION_CATEGORIES)}
# 创建整数 ID 到类别名称的映射字典 (用于将模型输出的数字 ID 转换回类别名称)
CATEGORY_TO_ID = {cat: i for i, cat in enumerate(PREDICTION_CATEGORIES)}
# 计算总的分类标签数量，这将决定模型输出层的维度
NUM_LABELS = len(PREDICTION_CATEGORIES)


# 为分类模型设置一个标准的最大序列长度
# 输入文本在分词后会填充或截断到这个长度。对于 BERT/RoBERTa 等模型，通常是 512
STANDARD_MAX_LENGTH = 512 # 保持512，如果后续显存仍然不足，可以适当减小（如384或256）

# 设置训练过程中保存模型检查点和日志的输出目录
OUTPUT_DIR = "./fine_tuned_model_output"
# 如果输出目录不存在，则创建它
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建输出目录: {OUTPUT_DIR}")

# --- 加载 Tokenizer ---
print(f"Loading tokenizer from {MODEL_PATH} using RobertaTokenizer...")
try:
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)

    model_max_length_info = getattr(tokenizer, 'model_max_length', 'N/A')
    print(f"Tokenizer model_max_length: {model_max_length_info}")
    if isinstance(model_max_length_info, int) and model_max_length_info > 10000:
         logging.warning(f"Tokenizer model_max_length ({model_max_length_info}) 异常大，将使用固定的 STANDARD_MAX_LENGTH ({STANDARD_MAX_LENGTH})")

    if tokenizer.pad_token_id is None:
        if '[PAD]' in tokenizer.vocab:
            tokenizer.pad_token_id = tokenizer.vocab['[PAD]']
        elif tokenizer.eos_token is not None and tokenizer.eos_token in tokenizer.vocab:
             tokenizer.pad_token_id = tokenizer.vocab[tokenizer.eos_token]
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token_id = tokenizer.vocab['[PAD]']
            print("Added [PAD] token to tokenizer.")
        print(f"Set pad_token_id to: {tokenizer.pad_token_id}")

except Exception as e:
    logging.error(f"加载 Tokenizer 时出错: {e}", exc_info=True)
    exit()


# --- 加载模型 (用于序列分类) ---
print(f"Loading model from {MODEL_PATH} using RobertaForSequenceClassification...")
try:
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=NUM_LABELS,
        id2label=ID_TO_CATEGORY,
        label2id=CATEGORY_TO_ID,
    )
    print("Model loaded successfully.")

except Exception as e:
    logging.error(f"加载模型时出错: {e}", exc_info=True)
    exit()


# --- 数据预处理 ---
train_hf_dataset = Dataset.from_list(train_data)
valid_hf_dataset = Dataset.from_list(valid_data)
test_hf_dataset = Dataset.from_list(test_data)

def preprocess_function(examples):
    encoding = tokenizer(
        examples['text'],
        add_special_tokens=True,
        max_length=STANDARD_MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True, # Depending on the model, this might be needed
    )

    # 将字符串标签列表转换为对应的整数 ID 列表
    labels = [CATEGORY_TO_ID.get(label_name, -1) for label_name in examples['label']]

    # 检查是否存在未知标签（标签 ID 为 -1）
    if -1 in labels:
        unknown_labels = [examples['label'][i] for i, label_id in enumerate(labels) if label_id == -1]
        logging.warning(f"数据集中发现未知标签: {set(unknown_labels)}。请确保所有标签都在 PREDICTION_CATEGORIES 中。这些样本的标签将被设置为 -100。")
        # 将未知标签 (-1) 转换为 -100。Trainer 会忽略标签为 -100 的样本。
        labels = [-100 if label_id == -1 else label_id for label_id in labels]

    encoding['labels'] = labels

    return encoding

train_tokenized_dataset = train_hf_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'label'])
valid_tokenized_dataset = valid_hf_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'label'])
test_tokenized_dataset = test_hf_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'label'])


print("数据集预处理完成。")
print(f"训练集样本数: {len(train_tokenized_dataset)}")
print(f"验证集样本数: {len(valid_tokenized_dataset)}")
print(f"测试集样本数: {len(test_tokenized_dataset)}")
if len(train_tokenized_dataset) > 0:
  print("第一个训练样本结构:")
  print(train_tokenized_dataset[0])


# --- 定义评估指标函数 ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # 过滤掉真实标签为 -100 的样本
    true_labels = labels[labels != -100]
    true_predictions = predictions[labels != -100]

    # 检查是否有有效样本进行评估
    if len(true_labels) == 0:
        logging.warning("评估集中没有有效样本 (标签非-100)，无法计算指标。")
        return {"accuracy": 0.0, "f1_macro": 0.0}

    accuracy_metric = load("accuracy")
    f1_metric = load("f1")

    accuracy = accuracy_metric.compute(predictions=true_predictions, references=true_labels)
    # 计算 macro F1 分数，忽略标签为 -100 的类别
    # 如果你的类别数量较少且分布不均，average="macro" 是一个不错的选择
    f1_score = f1_metric.compute(predictions=true_predictions, references=true_labels, average="macro")


    return {
        "accuracy": accuracy["accuracy"],
        "f1_macro": f1_score["f1"],
    }

# --- 设置训练参数 ---
# 定义训练的总轮数
TRAINING_EPOCHS = 5 # 或者您希望训练的实际轮数，例如 5 或 10

# 创建一个 TrainingArguments 对象，配置训练过程的各种参数
# *** 参数调整以适应 8GB 显存 ***
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, # 保存模型检查点和 Trainer 状态的目录
    num_train_epochs=TRAINING_EPOCHS, # 训练的总轮数
    per_device_train_batch_size=8, # *** 关键参数：每个设备上的训练批次大小，从8开始尝试 ***
    gradient_accumulation_steps=2, # *** 关键参数：增加梯度累积步数 ***
                                   # 有效批量大小 = per_device_train_batch_size * gradient_accumulation_steps
                                   # 这里有效批量大小 = 8 * 2 = 16。这是一个比较平衡的选择。
                                   # 如果 8 + 2 组合仍然 OOM，尝试 per_device_train_batch_size=4, gradient_accumulation_steps=4 (有效批量16)
                                   # 或者 per_device_train_batch_size=4, gradient_accumulation_steps=2 (有效批量8，显存占用更低)
    per_device_eval_batch_size=8,   # 评估批量大小，可以和训练批量大小相同，或者稍大一点尝试
    warmup_steps=100, # 预热步数
    weight_decay=0.01, # L2 正则化强度
    logging_dir=f"{OUTPUT_DIR}/logs", # TensorBoard 日志目录
    logging_steps=50, # 日志记录频率

    eval_strategy="epoch", # 每训练完一个 epoch 进行一次验证
    save_strategy="epoch", # *** 关键参数：每训练完一个 epoch 保存一次模型检查点 ***

    # === 修改开始：设置 save_total_limit 以保留每一轮的模型 ===
    # 将 save_total_limit 设置为大于或等于总训练轮数 TRAINING_EPOCHS 的值
    # 或者设置为 None 来保留所有检查点 (需要足够的磁盘空间)
    save_total_limit=TRAINING_EPOCHS, # 保留所有 epoch 的检查点
    # 如果您想确保绝对保留所有（包括可能的中间检查点），可以设置为 None 或一个更大的数
    # save_total_limit=None, # <-- 如果磁盘空间充足，可以尝试这个设置
    # === 修改结束 ===

    metric_for_best_model="f1_macro", # 指定用于判断最佳模型的指标
    greater_is_better=True, # 对于 "f1_macro" 指标，值越大越好
    fp16=True, # *** 开启混合精度训练 (Floating Point 16) *** 保持开启，显著减少显存
)

# --- 定义 Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=valid_tokenized_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer, # 将 tokenizer 传递给 Trainer，以便处理数据 collator
)

# --- 开始训练 ---
print("\n" + "="*20 + " 开始训练 " + "="*20 + "\n")
try:
    if not torch.cuda.is_available():
        logging.warning("未检测到 CUDA 可用设备，训练将使用 CPU，速度会非常慢！")

    # 调用 trainer.train() 方法开始训练过程
    train_result = trainer.train() # train() 方法返回一个 TrainOutput 对象
    print("\n" + "="*20 + " 训练完成 " + "="*20 + "\n")

    # --- 评估模型 ---
    # 这里的 evaluate 默认是在训练结束时的模型状态上进行的评估 (即最后一个 epoch 的模型)

    print("\n在验证集上评估训练结束时的模型:")
    eval_results_end = trainer.evaluate(eval_dataset=valid_tokenized_dataset)
    print(f"验证集评估结果 (训练结束): {eval_results_end}")

    print(f"\n在测试集上进行最终评估 (训练结束时的模型):")
    test_results_end = trainer.evaluate(eval_dataset=test_tokenized_dataset)
    print(f"测试集评估结果 (训练结束): {test_results_end}")


    # --- 模型已保存到 OUTPUT_DIR ---
    print(f"\n微调后的模型检查点已保存到目录: {OUTPUT_DIR}")
    print(f"根据 TrainingArguments 设置，每个 epoch 的检查点都已被保留。")
    # 您可以查看 OUTPUT_DIR 下的 checkpoint-XXXX 文件夹。
    # 要评估特定 epoch 的模型，需要手动加载对应的检查点。


except RuntimeError as e:
     # 专门捕获 RuntimeError，可能是 CUDA OOM 错误
     if "out of memory" in str(e).lower():
         logging.error("CUDA 显存不足 (Out of Memory) 错误！请尝试以下方法：")
         logging.error("1. 减小 per_device_train_batch_size (例如从 8 减到 4)。")
         logging.error("2. 增加 gradient_accumulation_steps (例如如果 batch size 是 4，可以增加到 4 或 8)。")
         logging.error(f"   当前设置: per_device_train_batch_size={training_args.per_device_train_batch_size}, gradient_accumulation_steps={training_args.gradient_accumulation_steps}")
         logging.error(f"   有效批量大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
         logging.error("3. 减小 STANDARD_MAX_LENGTH (例如从 512 减到 384 或 256)，但这可能影响模型性能。")
         logging.error("4. 检查是否有其他程序占用了 GPU 显存。")
         logging.error(f"原始错误信息: {e}", exc_info=True)
     else:
         # 其他 RuntimeErrors
         logging.error(f"训练过程中发生 RuntimeError: {e}", exc_info=True)
     print("\n" + "="*20 + " 训练发生 RuntimeError " + "="*20 + "\n")
except Exception as e:
    # 捕获其他所有异常
    logging.error(f"训练过程中发生未知错误: {e}", exc_info=True)
    print("\n" + "="*20 + " 训练发生未知错误 " + "="*20 + "\n")