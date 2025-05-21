# # app/utils/classification.py
# import logging
# import torch
# import torch.nn.functional as F
# import re
# import os
# from transformers import BertTokenizer, BertForSequenceClassification
# # from pathlib import Path # 如果用不到 Path 可以不导入
# from typing import Optional, List, Dict, Any
#
# # --- 从你原有代码中提取的规则导入 ---
# # 假设你的规则定义在 app/utils/rules.py.py 中
# # 如果在其他位置，请修改导入路径
# try:
#     from app.utils.rules import RULE_PATTERNS
# except ImportError as e:
#     logging.error(f"无法导入规则文件：{e}. 请检查 app/utils/rules.py 文件是否存在以及导入路径是否正确。", exc_info=True)
#     RULE_PATTERNS = {} # 如果导入失败，规则匹配将不可用
#
# # --- 从你原有代码中提取的模型加载配置 ---
# # 假设你的配置信息在 app/core/config.py 中，或者在这里直接定义路径
# # from app.core.config import settings # 如果使用 config.py
#
# logger = logging.getLogger(__name__)
#
# # 定义模型路径，请根据你的实际文件位置修改
# ORIGINAL_MODEL_PATH = r"D:\biyesheji\Biyesheji2.0\trainModel\fine_tuned_model_output\checkpoint-981"
# QUANTIZED_FP16_MODEL_PATH = r"D:\biyesheji\Biyesheji2.0\lianghua\quantized_models\model_fp16.pth"
#
# # 定义敏感信息类别列表，必须与训练模型时使用的标签类别完全一致
# SENSITIVE_CATEGORIES = [
#     "个人隐私信息",
#     "黄色信息",
#     "不良有害信息",
#     "无敏感信息"
# ]
# PREDICTION_CATEGORIES = SENSITIVE_CATEGORIES
#
# # 创建类别名称到整数 ID 的映射字典
# ID_TO_CATEGORY = {i: cat for i, cat in enumerate(PREDICTION_CATEGORIES)}
# CATEGORY_TO_ID = {cat: i for i, cat in enumerate(PREDICTION_CATEGORIES)}
# NUM_LABELS = len(PREDICTION_CATEGORIES)
#
# # 定义 BERT 模型处理的最大序列长度
# STANDARD_CLASSIFICATION_MAX_LENGTH = 512
#
# # --- 检测并设置推理设备 ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# logger.info(f"分类模型推理设备已设置为: {DEVICE}")
# # (此处省略设备信息日志，可放在应用启动时打印)
#
# # 初始化模型和 Tokenizer 变量
# bert_tokenizer: Optional[BertTokenizer] = None
# bert_model: Optional[BertForSequenceClassification] = None
# bert_loading_error: Optional[str] = None
#
# # --- 模型加载函数 (在应用启动时调用) ---
# async def load_classification_model():
#     global bert_tokenizer, bert_model, bert_loading_error
#
#     if bert_model is not None and bert_tokenizer is not None and bert_loading_error is None:
#         # logger.info("分类模型已加载，跳过重复加载。") # 避免频繁日志
#         return
#
#     logger.info(f"正在加载 BERT Tokenizer from {ORIGINAL_MODEL_PATH}...")
#     try:
#         bert_tokenizer = BertTokenizer.from_pretrained(ORIGINAL_MODEL_PATH)
#         if bert_tokenizer.pad_token_id is None:
#              bert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#              bert_tokenizer.pad_token_id = bert_tokenizer.vocab['[PAD]']
#              logger.info("Added [PAD] token.")
#
#     except Exception as e:
#         logger.error(f"加载 BERT Tokenizer 时出错: {e}", exc_info=True)
#         bert_loading_error = f"加载 BERT Tokenizer 时出错: {e}"
#         return
#
#     if bert_tokenizer is not None:
#         if not os.path.exists(QUANTIZED_FP16_MODEL_PATH):
#             logger.critical(f"量化后的 FP16 模型文件未找到：{QUANTIZED_FP16_MODEL_PATH}")
#             bert_loading_error = f"FP16 模型文件未找到：{QUANTIZED_FP16_MODEL_PATH}"
#             return
#         else:
#             logger.info(f"正在加载量化后的 FP16 BERT 分类模型 state_dict from {QUANTIZED_FP16_MODEL_PATH}...")
#             try:
#                 model_structure = BertForSequenceClassification.from_pretrained(
#                     ORIGINAL_MODEL_PATH,
#                     num_labels=NUM_LABELS,
#                     id2label=ID_TO_CATEGORY,
#                     label2id=CATEGORY_TO_ID,
#                     torch_dtype=torch.float32
#                 )
#                 state_dict = torch.load(QUANTIZED_FP16_MODEL_PATH, map_location=DEVICE)
#                 model_structure.load_state_dict(state_dict)
#
#                 bert_model = model_structure.half()
#                 bert_model.to(DEVICE)
#                 bert_model.eval()
#
#                 logger.info("量化后的 FP16 BERT 分类模型加载成功。")
#
#             except Exception as e:
#                 logger.error(f"加载量化后的 FP16 BERT 分类模型时出错: {e}", exc_info=True)
#                 bert_loading_error = f"加载 FP16 模型时出错: {e}"
#
#
# # ====================== 分类逻辑函数 (由服务层调用) ======================
#
# def classify_with_rules(text: str) -> Optional[str]:
#     """
#     使用预设的正则表达式规则对文本进行敏感信息分类。
#     """
#     if not RULE_PATTERNS:
#         return None
#     if not isinstance(text, str) or not text.strip():
#         return None
#     text_lower = text.lower()
#     for category, patterns in RULE_PATTERNS.items(): # 直接迭代字典，假设结构是 {category: [pattern1, pattern2]}
#         for pattern in patterns:
#             try:
#                 if pattern.search(text_lower):
#                     return category
#             except Exception as e:
#                  logger.error(f"规则匹配时发生错误，模式: {pattern.pattern[:50]}... 错误: {e}", exc_info=True)
#     return None
#
# def classify_text_with_model(text_to_classify: str) -> str:
#     """
#     使用加载的 BERT 分类模型对文本进行敏感信息分类。
#     """
#     if bert_model is None or bert_tokenizer is None:
#         # logger.error("分类模型或 Tokenizer 未加载，无法进行模型分类。")
#         return "分类失败"
#
#     if not isinstance(text_to_classify, str) or not text_to_classify.strip():
#          return "无敏感信息"
#
#     CONFIDENCE_THRESHOLD = 0.90 # 可从 config 读取
#
#     try:
#         inputs = bert_tokenizer(
#             text_to_classify,
#             return_tensors="pt",
#             truncation=True,
#             padding='max_length',
#             max_length=STANDARD_CLASSIFICATION_MAX_LENGTH
#         )
#         input_ids = inputs['input_ids'].to(DEVICE)
#         attention_mask = inputs['attention_mask'].to(DEVICE)
#         token_type_ids = inputs.get('token_type_ids')
#         if token_type_ids is not None:
#              token_type_ids = token_type_ids.to(DEVICE)
#
#         with torch.no_grad():
#             if token_type_ids is not None:
#                  outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#             else:
#                  outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
#
#         logits = outputs.logits[0]
#         probabilities = F.softmax(logits, dim=-1)
#         max_probability, predicted_class_id = torch.max(probabilities, dim=-1)
#         predicted_category = ID_TO_CATEGORY.get(predicted_class_id.item(), "分类失败")
#
#         if predicted_category != "无敏感信息" and max_probability.item() < CONFIDENCE_THRESHOLD:
#              return "无敏感信息"
#         elif predicted_category == "分类失败":
#              return "无敏感信息"
#         else:
#              return predicted_category
#
#     except Exception as e:
#         logger.error(f"BERT 分类模型推理过程中出错: {e}", exc_info=True)
#         return "分类失败"
#
# def classify_text(text: str) -> str:
#     """
#     结合规则匹配和 BERT 模型分类对文本进行敏感信息分类。
#     """
#     if not isinstance(text, str) or not text.strip():
#         return "无敏感信息"
#
#     rule_category = classify_with_rules(text)
#
#     if rule_category is not None:
#         return rule_category
#     else:
#         model_category = classify_text_with_model(text)
#         return model_category
#
# # --- 用于检查模型加载状态的依赖注入提供者 (可选，但推荐) ---
# # 如果你想让服务层或API层依赖于模型是否加载成功，可以定义一个依赖
# # 例如：
# # from fastapi import Depends, HTTPException
# # async def get_classification_resources():
# #     if bert_model is None or bert_tokenizer is None or bert_loading_error:
# #          raise HTTPException(status_code=503, detail=f"分类模型未加载成功: {bert_loading_error or '未知错误'}")
# #     return {"model": bert_model, "tokenizer": bert_tokenizer, "categories": ID_TO_CATEGORY}





# app/utils/classification.py
import logging
import torch
import torch.nn.functional as F
import re
import os
import sys
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # 确保 CUDA 上下文已初始化
import numpy as np

# from pathlib import Path # 如果用不到 Path 可以不导入
from typing import Optional, List, Dict, Any

# --- 从你原有代码中提取的规则导入 ---
# 假设你的规则定义在 app/utils/rules.py.py 中
# 如果在其他位置，请修改导入路径
try:
    from app.utils.rules import RULE_PATTERNS
except ImportError as e:
    logging.error(f"无法导入规则文件：{e}. 请检查 app/utils/rules.py 文件是否存在以及导入路径是否正确。", exc_info=True)
    RULE_PATTERNS = {} # 如果导入失败，规则匹配将不可用

# --- 从你原有代码中提取的模型加载配置 ---
# 假设你的配置信息在 app/core/config.py 中，或者在这里直接定义路径
# from app.core.config import settings # 如果使用 config.py

logger = logging.getLogger(__name__)

# 定义模型路径，请根据你的实际文件位置修改
# BERT Tokenizer 需要从原始模型路径加载，因为 TensorRT 引擎只包含推理图，不包含 tokenizer
ORIGINAL_MODEL_PATH = r"D:\biyesheji\Biyesheji2.0\trainModel\fine_tuned_model_output\checkpoint-981"
# TensorRT Engine 路径
TENSORRT_ENGINE_PATH = r"D:\biyesheji\Biyesheji2.0\lianghua\model_fp16_tensorrt_engine_batch_32.trt"


# 定义敏感信息类别列表，必须与训练模型时使用的标签类别完全一致
SENSITIVE_CATEGORIES = [
    "个人隐私信息",
    "黄色信息",
    "不良有害信息",
    "无敏感信息"
]
PREDICTION_CATEGORIES = SENSITIVE_CATEGORIES

# 创建类别名称到整数 ID 的映射字典
ID_TO_CATEGORY = {i: cat for i, cat in enumerate(PREDICTION_CATEGORIES)}
CATEGORY_TO_ID = {cat: i for i, cat in enumerate(PREDICTION_CATEGORIES)}
NUM_LABELS = len(PREDICTION_CATEGORIES)

# 定义 BERT 模型处理的最大序列长度
STANDARD_CLASSIFICATION_MAX_LENGTH = 512
# TensorRT 引擎预期的最大批次大小，应与构建引擎时使用的最大批次大小一致
TENSORRT_MAX_BATCH_SIZE = 32

# --- 检测并设置推理设备 ---
# TensorRT 总是使用 CUDA，因此这里确认 CUDA 是否可用
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    logger.warning("CUDA 不可用，TensorRT 引擎无法运行。请检查 PyTorch 和 CUDA 安装。")
    # 如果 CUDA 不可用，TensorRT 相关变量应设置为 None，防止后续错误
    trt_engine = None
    trt_context = None
    trt_inputs_info = None
    trt_outputs_info = None
    trt_bindings = None
    trt_stream = None
else:
    logger.info(f"分类模型推理设备已设置为: {DEVICE}")

# 初始化模型和 Tokenizer 变量
bert_tokenizer: Optional[Any] = None # 将是 transformers.BertTokenizer
trt_engine: Optional[trt.ICudaEngine] = None
trt_context: Optional[trt.IExecutionContext] = None
trt_inputs_info: Optional[List[Any]] = None
trt_outputs_info: Optional[List[Any]] = None
trt_bindings: Optional[List[int]] = None
trt_stream: Optional[cuda.Stream] = None
bert_loading_error: Optional[str] = None # 保留 bert_loading_error

# --- TensorRT Logger ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path: str):
    """反序列化并返回一个 TensorRT ICudaEngine 对象。"""
    if not os.path.exists(engine_path):
        logger.error(f"TensorRT 引擎文件未找到: {engine_path}")
        raise FileNotFoundError(f"TensorRT 引擎文件未找到: {engine_path}")

    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    logger.info(f"已从 {engine_path} 加载 TensorRT 引擎")
    return engine

def allocate_buffers(engine, max_batch_size: int, max_seq_len: int):
    """
    为每个张量绑定分配主机/设备缓冲区。
    `max_batch_size` 和 `max_seq_len` 用于确定最大可能的缓冲区大小。
    """
    inputs_info = []  # 存储 (名称, 主机内存, 设备内存, 原始形状)
    outputs_info = []  # 存储 (名称, 主机内存, 设备内存, 原始形状)

    total_tensors = engine.num_io_tensors
    bindings = [None] * total_tensors
    stream = cuda.Stream()  # CUDA 流

    for i in range(total_tensors):
        binding_name = engine.get_tensor_name(i)
        tensor_mode = engine.get_tensor_mode(binding_name)
        binding_shape = engine.get_tensor_shape(binding_name)
        binding_dtype = trt.nptype(engine.get_tensor_dtype(binding_name))

        binding_idx = i #engine.get_binding_index(binding_name) # 对于 TRT 8.x，使用 get_tensor_name 获取索引

        if tensor_mode == trt.TensorIOMode.INPUT:
            # 对于缓冲区分配，我们总是使用最大的可能维度
            resolved_shape_for_buffer = (max_batch_size, max_seq_len)
            size = int(np.prod(resolved_shape_for_buffer))

            host_mem = cuda.pagelocked_empty(size, binding_dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings[binding_idx] = int(dev_mem)
            inputs_info.append((binding_name, host_mem, dev_mem, binding_shape))

        elif tensor_mode == trt.TensorIOMode.OUTPUT:
            # 对于缓冲区分配，我们总是使用最大的可能维度
            # 分类的输出是 (批次大小, 标签数量)
            resolved_shape_for_buffer = (max_batch_size, NUM_LABELS)
            size = int(np.prod(resolved_shape_for_buffer))

            host_mem = cuda.pagelocked_empty(size, binding_dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings[binding_idx] = int(dev_mem)
            outputs_info.append((binding_name, host_mem, dev_mem, binding_shape))
        else:
            logger.warning(f"未知张量模式 for {binding_name}: {tensor_mode}")

    return inputs_info, outputs_info, bindings, stream


# --- 模型加载函数 (在应用启动时调用) ---
async def load_classification_model():
    global bert_tokenizer, trt_engine, trt_context, trt_inputs_info, trt_outputs_info, trt_bindings, trt_stream, bert_loading_error

    if trt_engine is not None and bert_tokenizer is not None and bert_loading_error is None:
        # logger.info("分类模型已加载，跳过重复加载。") # 避免频繁日志
        return

    if DEVICE == "cpu":
        bert_loading_error = "CUDA 不可用，TensorRT 引擎无法加载。"
        logger.error(bert_loading_error)
        return

    logger.info(f"正在加载 BERT Tokenizer from {ORIGINAL_MODEL_PATH}...")
    try:
        from transformers import BertTokenizer # 在这里导入以避免如果此文件被提前导入时的循环依赖
        bert_tokenizer = BertTokenizer.from_pretrained(ORIGINAL_MODEL_PATH)
        if bert_tokenizer.pad_token_id is None:
             # BertTokenizer 通常有 pad_token_id，如果没有，手动添加
             bert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             bert_tokenizer.pad_token_id = bert_tokenizer.vocab['[PAD]']
             logger.info("已为 tokenizer 添加 [PAD] token。")

    except Exception as e:
        logger.error(f"加载 BERT Tokenizer 时出错: {e}", exc_info=True)
        bert_loading_error = f"加载 BERT Tokenizer 时出错: {e}"
        return

    if bert_tokenizer is not None:
        if not os.path.exists(TENSORRT_ENGINE_PATH):
            logger.critical(f"TensorRT 引擎文件未找到：{TENSORRT_ENGINE_PATH}")
            bert_loading_error = f"TensorRT 引擎文件未找到：{TENSORRT_ENGINE_PATH}"
            return
        else:
            logger.info(f"正在加载 TensorRT 引擎 from {TENSORRT_ENGINE_PATH}...")
            try:
                trt_engine = load_engine(TENSORRT_ENGINE_PATH)
                trt_context = trt_engine.create_execution_context()
                trt_inputs_info, trt_outputs_info, trt_bindings, trt_stream = \
                    allocate_buffers(trt_engine, TENSORRT_MAX_BATCH_SIZE, STANDARD_CLASSIFICATION_MAX_LENGTH)

                logger.info("TensorRT 引擎和缓冲区加载成功。")

                # 打印 TensorRT 引擎的输出绑定名称以帮助调试
                logger.info("TensorRT 引擎输出绑定名称:")
                for i in range(trt_engine.num_io_tensors):
                    if trt_engine.get_tensor_mode(trt_engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT:
                        logger.info(f"- {trt_engine.get_tensor_name(i)}")
                logger.info(f"ID_TO_CATEGORY 映射: {ID_TO_CATEGORY}")


            except Exception as e:
                logger.error(f"加载 TensorRT 引擎时出错: {e}", exc_info=True)
                bert_loading_error = f"加载 TensorRT 引擎时出错: {e}"


# ====================== 分类逻辑函数 (由服务层调用) ======================

def classify_with_rules(text: str) -> Optional[str]:
    """
    使用预设的正则表达式规则对文本进行敏感信息分类。
    """
    if not RULE_PATTERNS:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    text_lower = text.lower()
    for category, patterns in RULE_PATTERNS.items(): # 直接迭代字典，假设结构是 {category: [pattern1, pattern2]}
        for pattern in patterns:
            try:
                if pattern.search(text_lower):
                    return category
            except Exception as e:
                 logger.error(f"规则匹配时发生错误，模式: {pattern.pattern[:50]}... 错误: {e}", exc_info=True)
    return None

def classify_text_with_model(text_to_classify: str) -> str:
    """
    使用加载的 TensorRT 分类模型对文本进行敏感信息分类。
    """
    global trt_context, trt_inputs_info, trt_outputs_info, trt_bindings, trt_stream

    if trt_engine is None or bert_tokenizer is None or trt_context is None:
        logger.error("分类模型或 Tokenizer 未加载，无法进行模型分类。")
        return "分类失败"

    if not isinstance(text_to_classify, str) or not text_to_classify.strip():
         return "无敏感信息"

    CONFIDENCE_THRESHOLD = 0.90 # 可从 config 读取

    try:
        # 对输入文本进行 Tokenize
        inputs = bert_tokenizer(
            text_to_classify,
            return_tensors="np", # 返回 NumPy 数组，供 TensorRT 使用
            truncation=True,
            padding='max_length',
            max_length=STANDARD_CLASSIFICATION_MAX_LENGTH
        )

        input_ids_np = inputs['input_ids'].astype(np.int32)
        attention_mask_np = inputs['attention_mask'].astype(np.int32)
        token_type_ids_np = inputs.get('token_type_ids', np.zeros_like(input_ids_np)).astype(np.int32)

        # 重塑为单次推理 (批次大小为 1)
        input_ids_np = input_ids_np.reshape(1, STANDARD_CLASSIFICATION_MAX_LENGTH)
        attention_mask_np = attention_mask_np.reshape(1, STANDARD_CLASSIFICATION_MAX_LENGTH)
        token_type_ids_np = token_type_ids_np.reshape(1, STANDARD_CLASSIFICATION_MAX_LENGTH)


        # 设置动态输入形状并复制数据
        current_batch_size = 1 # 我们一次只推理一个文本

        for input_info_item in trt_inputs_info:
            input_name, host_mem, dev_mem, original_binding_shape = input_info_item

            new_shape = (current_batch_size, STANDARD_CLASSIFICATION_MAX_LENGTH)
            trt_context.set_input_shape(input_name, new_shape)

            if input_name == "input_ids":
                np.copyto(host_mem[:input_ids_np.size], input_ids_np.ravel())
            elif input_name == "attention_mask":
                np.copyto(host_mem[:attention_mask_np.size], attention_mask_np.ravel())
            elif input_name == "token_type_ids":
                np.copyto(host_mem[:token_type_ids_np.size], token_type_ids_np.ravel())
            else:
                logger.warning(f"TensorRT 推理的未知输入名称 '{input_name}'。跳过数据复制。")
                continue

            cuda.memcpy_htod_async(dev_mem, host_mem, trt_stream)

        # 执行推理
        trt_context.execute_v2(bindings=trt_bindings)
        trt_stream.synchronize() # 执行后同步流

        # 获取输出
        logits_output_info = None
        for output_info_item in trt_outputs_info:
            output_name, host_mem, dev_mem, _ = output_info_item
            if 'output' in output_name.lower() or 'logits' in output_name.lower(): # 假设 logits 输出名称
                logits_output_info = output_info_item
                break

        if logits_output_info is None:
            logger.error("在 TensorRT 引擎中找不到 logits 输出绑定。请检查引擎构建时的输出节点名称。")
            return "分类失败"

        host_mem_for_output = logits_output_info[1]
        dev_mem_for_output = logits_output_info[2]

        cuda.memcpy_dtoh_async(host_mem_for_output, dev_mem_for_output, trt_stream)
        trt_stream.synchronize() # 确保 D2H 复制完成

        actual_output_shape = trt_context.get_tensor_shape(logits_output_info[0])
        # 重塑为 (批次大小, 标签数量)
        logits = host_mem_for_output[:np.prod(actual_output_shape)].reshape(actual_output_shape)

        # 应用 softmax 并获取预测
        logits_torch = torch.from_numpy(logits) # 使用一个新变量名，避免混淆
        probabilities = F.softmax(logits_torch, dim=-1) # 使用转换后的 torch 张量
        max_probability, predicted_class_id = torch.max(probabilities, dim=-1)

        # --- 调试日志 ---
        logger.info(f"模型预测原始结果： ID={predicted_class_id.item()}, 概率={max_probability.item():.4f}")
        # --- 调试日志结束 ---

        predicted_category = ID_TO_CATEGORY.get(predicted_class_id.item(), "分类失败")

        if predicted_category == "分类失败":
             logger.warning(f"ID_TO_CATEGORY 映射失败或预测 ID ({predicted_class_id.item()}) 为未知。最终返回 '无敏感信息'。")
             return "无敏感信息" # 如果直接从模型输出就是“分类失败”，则视为无敏感信息

        if predicted_category != "无敏感信息" and max_probability.item() < CONFIDENCE_THRESHOLD:
             logger.info(f"分类结果 '{predicted_category}' 因置信度 ({max_probability.item():.4f}) 低于阈值 ({CONFIDENCE_THRESHOLD}) 而修正为 '无敏感信息'。")
             return "无敏感信息"
        else:
             logger.info(f"最终分类结果: {predicted_category}, 概率: {max_probability.item():.4f}")
             return predicted_category

    except Exception as e:
        logger.error(f"TensorRT 分类模型推理过程中出错: {e}", exc_info=True)
        return "分类失败"

def classify_text(text: str) -> str:
    """
    结合规则匹配和 TensorRT 模型分类对文本进行敏感信息分类。
    """
    if not isinstance(text, str) or not text.strip():
        return "无敏感信息"

    rule_category = classify_with_rules(text)

    if rule_category is not None:
        return rule_category
    else:
        model_category = classify_text_with_model(text)
        return model_category

# --- 用于检查模型加载状态的依赖注入提供者 (可选，但推荐) ---
# 如果你想让服务层或API层依赖于模型是否加载成功，可以定义一个依赖
# 例如：
# from fastapi import Depends, HTTPException
# async def get_classification_resources():
#     if trt_engine is None or bert_tokenizer is None or bert_loading_error:
#          raise HTTPException(status_code=503, detail=f"分类模型未加载成功: {bert_loading_error or '未知错误'}")
#     return {"engine": trt_engine, "tokenizer": bert_tokenizer, "categories": ID_TO_CATEGORY}



