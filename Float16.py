import torch
from transformers import BertForSequenceClassification, BertTokenizer
import os
import logging

# 设置日志，方便查看过程信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 替换为你的最佳检查点路径
MODEL_PATH = r"D:\biyesheji\ImageRecognition\trainModel\fine_tuned_model_output\checkpoint-981"
# 确保这是你训练好的、基于 4 类的模型
NUM_LABELS = 4

# 加载 Tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    logging.info("Tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading tokenizer: {e}")
    exit()

# 加载原始的 Float32 模型，用于 FP16 分支
try:
    model_fp32 = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_LABELS)
    model_fp32.eval()
    logging.info(f"Original FP32 model loaded successfully with {NUM_LABELS} labels.")
except Exception as e:
    logging.error(f"Error loading FP32 model: {e}", exc_info=True)
    exit()

# 定义保存量化模型的目录
QUANTIZED_MODEL_DIR = "./quantized_models"
os.makedirs(QUANTIZED_MODEL_DIR, exist_ok=True)

print("-" * 30)

# --- 方案 1：Float16 (FP16) 量化 ---
print("Starting Float16 (FP16) quantization...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    logging.info(f"Using GPU for FP16 quantization: {torch.cuda.get_device_name(0)}")
else:
    logging.warning("CUDA not available, FP16 quantization will run on CPU (速度可能较慢).")

try:
    # 只把 model_fp32 转半精度，且存为新变量 model_fp16
    model_fp32.to(DEVICE)
    model_fp16 = model_fp32.half()
    model_fp16.eval()
    logging.info("FP16 quantization applied successfully.")

    # 保存 FP16 state_dict
    fp16_model_path = os.path.join(QUANTIZED_MODEL_DIR, "model_fp16.pth")
    torch.save(model_fp16.state_dict(), fp16_model_path)
    size_fp16 = os.path.getsize(fp16_model_path) / (1024 * 1024)
    logging.info(f"FP16 model state_dict saved to {fp16_model_path}, size: {size_fp16:.2f} MB")

    print("Float16 quantization completed.")
except Exception as e:
    logging.error(f"Error during FP16 quantization: {e}", exc_info=True)

print("-" * 30)

# --- 方案 2：Dynamic Int8 量化 ---
print("Starting Dynamic Int8 quantization...")
try:
    # 重新加载一个全新的 FP32 模型，用于 Int8 量化
    model_fp32_for_int8 = BertForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=NUM_LABELS
    )
    model_fp32_for_int8.eval().to('cpu')

    # 对纯 FP32 模型做动态量化
    model_int8_dynamic = torch.quantization.quantize_dynamic(
        model_fp32_for_int8,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    model_int8_dynamic.eval()
    logging.info("Dynamic Int8 quantization applied successfully.")

    # 保存 Int8 模型 state_dict
    int8_model_path = os.path.join(QUANTIZED_MODEL_DIR, "model_int8_dynamic.pth")
    torch.save(model_int8_dynamic.state_dict(), int8_model_path)
    size_int8 = os.path.getsize(int8_model_path) / (1024 * 1024)
    logging.info(f"Dynamic Int8 model state_dict saved to {int8_model_path}, size: {size_int8:.2f} MB")

    print("Dynamic Int8 quantization completed.")
except Exception as e:
    logging.error(f"Error during Dynamic Int8 quantization: {e}", exc_info=True)

print("-" * 30)
