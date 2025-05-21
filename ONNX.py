import torch
from transformers import BertForSequenceClassification, BertTokenizer
import os

# --- 重要提示：这些代码通常会来自你的主脚本，即你加载 tokenizer 和 FP16 模型的地方 ---
# 示例（如果你要单独运行此部分，请替换为你的实际路径/加载逻辑）：
MODEL_PATH = r"D:\biyesheji\ImageRecognition\trainModel\fine_tuned_model_output\checkpoint-981"
NUM_LABELS = 4 # 或者你实际的标签数量

# 加载 Tokenizer（确保可用）
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    print("Tokenizer 已成功加载，用于 ONNX 导出。")
except Exception as e:
    print(f"加载 Tokenizer 时出错，无法导出 ONNX：{e}")
    exit()

# 加载 FP16 模型（确保可用）
# 你应该像在对比脚本中加载 FP16 模型那样加载它
# 例如：model_fp16 = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_LABELS).half().eval().to('cuda')
# 为简单起见，如果你单独运行此部分，这里提供一个占位符：
try:
    model_fp32_base = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_LABELS)
    # 确保它在 CUDA 上并且已转换为半精度（如果你想导出 FP16 ONNX）
    DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_fp16 = model_fp32_base.half().eval().to(DEVICE_GPU)
    print("FP16 模型已成功加载，用于 ONNX 导出。")
except Exception as e:
    print(f"加载 FP16 模型时出错，无法导出 ONNX：{e}")
    exit()
# --- 重要提示结束 ---

# 创建一个虚拟输入，模拟真实的输入形状和类型
# 注意：input_ids 和 token_type_ids 必须是 long 类型
# attention_mask 应该是 float32 类型
sample_text = "这是一段用于测试模型推理速度的文本，长度要足够，以模拟真实情况。"
dummy_input = tokenizer(
    sample_text,
    return_tensors="pt",
    max_length=512, # 使用你的 STANDARD_MAX_LENGTH
    truncation=True,
    padding="max_length"
)

# 将虚拟输入移动到与模型相同的设备
dummy_input = {k: v.to(model_fp16.device) for k, v in dummy_input.items()}

# 重要：调整数据类型以保持 ONNX 导出的一致性
# input_ids 和 token_type_ids 必须是 Long 类型
# attention_mask 应该是 Float 类型用于注意力机制
dummy_input['input_ids'] = dummy_input['input_ids'].long()
dummy_input['attention_mask'] = dummy_input['attention_mask'].float()
dummy_input['token_type_ids'] = dummy_input['token_type_ids'].long()

onnx_path = "./model_fp16.onnx"
# 确保目录存在
os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)

try:
    torch.onnx.export(
        model_fp16,
        # 以元组形式提供输入，顺序与模型预期的一致 (input_ids, attention_mask, token_type_ids)
        args=(dummy_input['input_ids'], dummy_input['attention_mask'], dummy_input['token_type_ids']),
        f=onnx_path,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['logits'],
        # 定义动态轴以实现灵活的批处理大小和序列长度
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                      'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                      'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
                      'logits': {0: 'batch_size'}},
        opset_version=14, # 使用 opset_version 14 通常是一个很好的平衡点。
                          # 更高的版本可能支持更多算子，但需要更新的 ONNX Runtime/TensorRT。
        do_constant_folding=True, # 通过预先计算常量表达式来提高性能
        # enable_experimental_op_fallback=True # 可能对自定义算子有用，但对于 BERT 通常不需要
    )
    print(f"FP16 模型已成功导出到 ONNX：{onnx_path}")
except Exception as e:
    print(f"ONNX 导出时出错：{e}")