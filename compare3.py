import os
import sys
import time
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Ensures CUDA context is initialized
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification  # 仍然需要BertForSequenceClassification用于PyTorch模型
from sklearn.metrics import f1_score, classification_report
import numpy as np
import logging
from tqdm.auto import tqdm
from types import SimpleNamespace  # For TRT output wrapper

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants Configuration ---
# Path to the fine-tuned Hugging Face model (used as base for FP32, and for tokenizer)
MODEL_PATH = r"D:\biyesheji\ImageRecognition\trainModel\fine_tuned_model_output\checkpoint-981"
# Path to the compiled TensorRT engine
ENGINE_PATH = r"D:\biyesheji\Biyesheji2.0\lianghua\model_fp16_tensorrt_engine_batch_32.trt"  # 已更新为你的路径
# Path to a saved PyTorch FP16 model's state_dict (optional, for comparison)
PYTORCH_FP16_MODEL_PATH = "./quantized_models/model_fp16.pth"

BATCH_SIZE = 32
STANDARD_MAX_LENGTH = 512

SENSITIVE_CATEGORIES = [
    "个人隐私信息",
    "黄色信息",
    "不良有害信息",
    "无敏感信息"
]
ID_TO_CATEGORY = {i: cat for i, cat in enumerate(SENSITIVE_CATEGORIES)}
CATEGORY_TO_ID = {cat: i for i, cat in enumerate(SENSITIVE_CATEGORIES)}
NUM_LABELS = len(SENSITIVE_CATEGORIES)

# --- Check TensorRT Version ---
logger.info(f"Detected TensorRT version: {trt.__version__}")
if int(trt.__version__.split('.')[0]) < 8:
    logger.warning(
        "Your TensorRT version is older than 8.x. The API used in this script might not be fully compatible.")
    logger.warning("Please consider upgrading TensorRT for optimal compatibility.")

# --- Data Import ---
try:
    data_dir = r'D:\biyesheji\ImageRecognition\trainModel'
    if data_dir not in sys.path:
        sys.path.append(data_dir)
    from train_data import train_data
    from valid_data import valid_data
    from test_data import test_data as raw_test_data

    logging.info("Successfully imported train, valid, and test dataset lists.")
except ImportError as e:
    logging.error(f"Failed to import dataset files: {e}")
    logging.error(f"Please ensure `train_data.py`, `valid_data.py`, and `test_data.py` exist in {data_dir},")
    logging.error(f"and each contains a Python list variable (e.g., `train_data = [...]`).")
    sys.exit(1)


# --- 1. Dataset Preparation ---
class TextClassificationDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_len):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = str(item['text'])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten().to(torch.float32),
            'token_type_ids': encoding['token_type_ids'].flatten().to(torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_and_preprocess_test_data(tokenizer, max_len, batch_size_dynamic):
    processed_labels = [CATEGORY_TO_ID.get(item['label'], -1) for item in raw_test_data]
    filtered_data, filtered_labels = [], []
    unknown_labels = set()

    for i, lid in enumerate(processed_labels):
        if lid != -1:
            filtered_data.append(raw_test_data[i])
            filtered_labels.append(lid)
        else:
            unknown_labels.add(raw_test_data[i]['label'])

    if unknown_labels:
        logger.warning(f"Unknown labels found in test set: {unknown_labels}. These samples will be filtered.")

    if not filtered_data:
        logger.error("No samples available for testing after preprocessing. Please check test data and labels.")
        sys.exit(1)

    dataset = TextClassificationDataset(filtered_data, filtered_labels, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size_dynamic, shuffle=False, num_workers=0)
    return loader, dataset


# --- 2. TensorRT Engine Utilities and Model Class (UPDATED) ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_trt_engine(engine_path: str):
    if not os.path.exists(engine_path):
        logger.error(f"TensorRT engine file not found: {engine_path}")
        raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    logger.info(f"Loaded TensorRT engine from {engine_path}")
    return engine


class TensorRTModel:
    def __init__(self, engine_path: str, max_batch_size: int, max_seq_len: int, num_labels_config: int):
        self.engine = load_trt_engine(engine_path)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_labels = num_labels_config
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Initialize attributes BEFORE calling _allocate_buffers
        self.inputs_info = []
        self.outputs_info = []
        self.bindings = [None] * self.engine.num_io_tensors
        self.output_binding_name = None  # Added/Moved this initialization

        self._allocate_buffers()

    def _allocate_buffers(self):
        # inputs_info, outputs_info, bindings are already initialized in __init__
        # self.inputs_info = [] # Removed redundant initialization
        # self.outputs_info = [] # Removed redundant initialization
        # self.bindings = [None] * self.engine.num_io_tensors # Removed redundant initialization

        for i in range(self.engine.num_io_tensors):
            binding_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(binding_name)
            binding_shape = self.engine.get_tensor_shape(binding_name)
            binding_dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))

            # Determine buffer_shape based on tensor mode and known max dimensions
            if tensor_mode == trt.TensorIOMode.INPUT:
                # For input buffers, we use the max sequence length as allocated size
                buffer_shape = (self.max_batch_size, self.max_seq_len)

                host_mem = cuda.pagelocked_empty(buffer_shape, binding_dtype)
                dev_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings[i] = int(dev_mem)
                self.inputs_info.append((binding_name, host_mem, dev_mem, binding_shape, binding_dtype))
                logger.debug(
                    f"Allocated TRT Input: {binding_name}, Buffer Shape: {buffer_shape}, Dtype: {binding_dtype}, Original Binding Shape: {binding_shape}")
            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                buffer_shape = (self.max_batch_size, self.num_labels)
                host_mem = cuda.pagelocked_empty(buffer_shape, binding_dtype)
                dev_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings[i] = int(dev_mem)
                self.outputs_info.append((binding_name, host_mem, dev_mem, binding_shape, binding_dtype))
                logger.debug(
                    f"Allocated TRT Output: {binding_name}, Buffer Shape: {buffer_shape}, Dtype: {binding_dtype}, Original Binding Shape: {binding_shape}")

                # Heuristically set output_binding_name if not already set
                if self.output_binding_name is None and (
                        'output' in binding_name.lower() or 'logits' in binding_name.lower()):
                    self.output_binding_name = binding_name
            else:
                logger.warning(f"Unknown tensor mode for {binding_name}: {tensor_mode}")

        if self.output_binding_name is None and self.outputs_info:
            self.output_binding_name = self.outputs_info[0][0]
            logger.warning(
                f"Could not heuristically determine logits output name. Defaulting to: {self.output_binding_name}")
        elif not self.outputs_info:
            raise RuntimeError("No output bindings found in the TensorRT engine.")

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None):
        current_batch_size = input_ids.size(0)
        current_seq_len = input_ids.size(1)  # Get actual sequence length for dynamic input

        # Prepare inputs for TRT
        input_data_map = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
            # Token type IDs might be optional for some models or have different names
            "token_type_ids": token_type_ids.cpu().numpy() if token_type_ids is not None else None
        }

        for name, host_mem, dev_mem, original_binding_shape, dtype in self.inputs_info:
            # Set dynamic shape for the current batch and sequence length
            # If original_binding_shape contains -1 for batch or sequence, update context
            if original_binding_shape[0] == -1 or original_binding_shape[1] == -1:
                # The shape for the context must match the actual shape of the input data
                self.context.set_input_shape(name, (current_batch_size, current_seq_len))

            data_np_original = None
            if "input_ids" in name.lower():
                data_np_original = input_data_map["input_ids"]
            elif "attention_mask" in name.lower():
                data_np_original = input_data_map["attention_mask"]
            elif "token_type_ids" in name.lower():
                data_np_original = input_data_map["token_type_ids"]

            if data_np_original is None:
                logger.error(f"Cannot map input binding '{name}' to provided input tensors.")
                raise ValueError(f"Unhandled input binding: {name}")

            # Ensure data type matches what the engine expects
            data_np = data_np_original.astype(dtype, copy=False)

            # Copy data to page-locked host memory, considering the actual shape
            # Use data_np.nbytes to copy the exact amount of data for the current batch
            cuda.memcpy_htod_async(dev_mem, data_np.tobytes(), self.stream)  # Use tobytes for byte copy

        # Execute model
        self.context.execute_v2(bindings=self.bindings)  # 注意：这里不再需要 stream_handle 参数

        # Retrieve output
        output_host_mem = None
        output_dev_mem = None

        # Find the correct output for logits
        for name, host_mem_o, dev_mem_o, _, _ in self.outputs_info:
            if name == self.output_binding_name:  # Use the identified output binding name
                output_host_mem = host_mem_o
                output_dev_mem = dev_mem_o
                break

        if output_host_mem is None:
            raise RuntimeError(f"Could not find output binding named '{self.output_binding_name}' or suitable default.")

        # Copy output from device to host
        cuda.memcpy_dtoh_async(output_host_mem, output_dev_mem, self.stream)
        self.stream.synchronize()  # Synchronize to ensure D2H copy is complete

        # Reshape and convert to tensor
        actual_output_shape_from_context = self.context.get_tensor_shape(self.output_binding_name)
        # Ensure only the relevant part of the buffer is used for the current batch size
        num_elements_output = np.prod(actual_output_shape_from_context)
        logits_np = output_host_mem.reshape(-1)[:num_elements_output].reshape(actual_output_shape_from_context)

        logits_torch = torch.from_numpy(logits_np.copy()).to("cpu")

        return SimpleNamespace(logits=logits_torch)

    def eval(self):
        pass

    def to(self, device):
        if str(device) != "cuda" and torch.cuda.is_available():
            logger.warning("TensorRTModel is designed for CUDA execution. Non-CUDA device specified in to().")
        return self


# --- 3. Main Workflow ---
def main():
    logger.info("--- Initializing ---")
    DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_CPU = torch.device("cpu")
    logger.info(f"Using GPU: {DEVICE_GPU}, CPU: {DEVICE_CPU}")

    # 3.1 Load Tokenizer
    logger.info("Loading Tokenizer...")
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        logger.info("Tokenizer loaded.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {MODEL_PATH}: {e}")
        sys.exit(1)

    # 3.2 Load Test Set
    logger.info("Loading and preprocessing test data...")
    test_loader, test_dataset = load_and_preprocess_test_data(tokenizer, STANDARD_MAX_LENGTH, BATCH_SIZE)
    logger.info(f"Test data loaded: {len(test_dataset)} samples, {len(test_loader)} batches.")
    if len(test_dataset) == 0:
        logger.error("No test samples loaded. Exiting.")
        sys.exit(1)

    # 3.3 Load Models
    logger.info("--- Loading Models for Comparison ---")
    models_to_compare = {}

    # PyTorch FP32 Model
    try:
        model_fp32 = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_LABELS)
        model_fp32.eval()
        models_to_compare["PyTorch FP32"] = model_fp32.to(DEVICE_GPU)
        logger.info("PyTorch FP32 model loaded.")
    except Exception as e:
        logger.error(f"Failed to load PyTorch FP32 model: {e}")
        models_to_compare["PyTorch FP32"] = None

    # PyTorch FP16 Model (Optional)
    if os.path.exists(PYTORCH_FP16_MODEL_PATH):
        try:
            # Load to CPU first, then convert to half and move to GPU
            model_fp16 = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_LABELS)
            model_fp16.load_state_dict(torch.load(PYTORCH_FP16_MODEL_PATH, map_location='cpu'))
            model_fp16.half()  # Convert to FP16
            model_fp16.eval()
            models_to_compare["PyTorch FP16"] = model_fp16.to(DEVICE_GPU)
            logger.info(f"PyTorch FP16 model loaded from {PYTORCH_FP16_MODEL_PATH}.")
        except Exception as e:
            logger.warning(f"Failed to load PyTorch FP16 model from {PYTORCH_FP16_MODEL_PATH}: {e}. Skipping.")
            models_to_compare["PyTorch FP16"] = None
    else:
        logger.info(f"PyTorch FP16 model state_dict not found at {PYTORCH_FP16_MODEL_PATH}. Skipping.")
        models_to_compare["PyTorch FP16"] = None

    # TensorRT FP16 Model
    if os.path.exists(ENGINE_PATH):
        try:
            # Pass BATCH_SIZE and STANDARD_MAX_LENGTH to TensorRTModel
            model_tensorrt_fp16 = TensorRTModel(ENGINE_PATH, BATCH_SIZE, STANDARD_MAX_LENGTH, NUM_LABELS)
            models_to_compare["TensorRT FP16"] = model_tensorrt_fp16
            logger.info("TensorRT FP16 model loaded.")
        except Exception as e:
            logger.error(f"Failed to load TensorRT FP16 model: {e}")
            models_to_compare["TensorRT FP16"] = None
    else:
        logger.warning(f"TensorRT engine not found at {ENGINE_PATH}. Skipping TensorRT FP16 model.")
        models_to_compare["TensorRT FP16"] = None

    # --- 4. Model Size Comparison ---
    logger.info("\n" + "=" * 20 + " Model Size Comparison " + "=" * 20)
    for name, model_instance in models_to_compare.items():
        if model_instance is None:
            logger.info(f"{name}: Not loaded, skipping size calculation.")
            continue
        if "TensorRT" in name:
            try:
                size_mb = os.path.getsize(ENGINE_PATH) / (1024 * 1024)
                logger.info(f"{name} engine file size: {size_mb:.2f} MB")
            except Exception as e:
                logger.error(f"Could not get size for {name}: {e}")
        else:  # PyTorch models
            try:
                temp_path = f"./temp_{name.replace(' ', '_').lower()}.pth"
                torch.save(model_instance.state_dict(), temp_path)
                size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                os.remove(temp_path)
                logger.info(f"{name} state_dict size: {size_mb:.2f} MB")
            except Exception as e:
                logger.error(f"Could not get state_dict size for {name}: {e}")
    logger.info("-" * 50)

    # --- 5. Inference Speed Comparison ---
    logger.info("\n" + "=" * 20 + " Inference Speed Comparison " + "=" * 20)
    sample_texts_for_speed = ["这是一个用于测试模型推理速度的示例文本。" * 10] * BATCH_SIZE
    try:
        speed_test_input = tokenizer(
            sample_texts_for_speed,
            return_tensors="pt",
            max_length=STANDARD_MAX_LENGTH,
            padding="max_length",
            truncation=True
        )
    except Exception as e:
        logger.error(f"Failed to tokenize sample texts for speed test: {e}")
        sys.exit(1)

    num_warmup_runs = 5
    num_timing_runs = 50

    for name, model_instance in models_to_compare.items():
        if model_instance is None:
            logger.info(f"{name}: Not loaded, skipping speed test.")
            continue

        logger.info(f"Testing speed for {name} (Batch Size: {BATCH_SIZE})...")
        current_device = DEVICE_GPU
        if not isinstance(model_instance, TensorRTModel):
            model_instance.to(current_device)
            model_instance.eval()

        if not isinstance(model_instance, TensorRTModel):
            inputs_for_speed = {k: v.to(current_device) for k, v in speed_test_input.items()}
        else:
            inputs_for_speed = {k: v.clone() for k, v in speed_test_input.items()}

        timings = []
        try:
            with torch.no_grad():
                # Warm-up runs
                for _ in range(num_warmup_runs):
                    _ = model_instance(
                        inputs_for_speed['input_ids'],
                        inputs_for_speed['attention_mask'],
                        inputs_for_speed.get('token_type_ids')
                    )
                    if isinstance(model_instance, TensorRTModel):
                        model_instance.stream.synchronize()
                    elif current_device.type == 'cuda':
                        torch.cuda.synchronize()

                # Timing runs
                for _ in tqdm(range(num_timing_runs), desc=f"Timing {name}", leave=False):
                    start_time = time.perf_counter()
                    _ = model_instance(
                        inputs_for_speed['input_ids'],
                        inputs_for_speed['attention_mask'],
                        inputs_for_speed.get('token_type_ids')
                    )
                    if isinstance(model_instance, TensorRTModel):
                        model_instance.stream.synchronize()
                    elif current_device.type == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    timings.append(end_time - start_time)

            avg_time_ms = (sum(timings) / len(timings)) * 1000
            logger.info(f"{name}: Average inference time = {avg_time_ms:.2f} ms/batch (Batch Size: {BATCH_SIZE})")
        except Exception as e:
            logger.error(f"Error during speed test for {name}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("-" * 50)

    # --- 6. Accuracy Evaluation ---
    logger.info("\n" + "=" * 20 + " Accuracy Evaluation " + "=" * 20)

    def evaluate_model_accuracy(model, loader, model_name_tag, device_for_eval):
        if model is None:
            logger.info(f"{model_name_tag}: Not loaded, skipping accuracy evaluation.")
            return {"f1": 0.0, "report": "N/A (Model not loaded)"}

        logger.info(f"Evaluating accuracy for {model_name_tag} on {device_for_eval}...")
        if not isinstance(model, TensorRTModel):
            model.to(device_for_eval)
            model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Eval {model_name_tag}", leave=False)):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                token_type_ids = batch.get('token_type_ids')
                labels = batch['labels']

                if not isinstance(model, TensorRTModel):
                    input_ids = input_ids.to(device_for_eval)
                    attention_mask = attention_mask.to(device_for_eval)
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(device_for_eval)

                outputs = model(input_ids, attention_mask, token_type_ids)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if not all_labels:
            logger.warning(f"No labels collected for {model_name_tag}. Skipping report.")
            return {"f1": 0.0, "report": "N/A (No labels)"}

        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        report = classification_report(all_labels, all_preds, target_names=SENSITIVE_CATEGORIES, zero_division=0,
                                       digits=4)
        return {"f1": f1, "report": report}

    for name, model_instance in models_to_compare.items():
        eval_device = DEVICE_GPU
        accuracy_result = evaluate_model_accuracy(model_instance, test_loader, name, eval_device)
        logger.info(f"\n--- {name} Accuracy ---")
        logger.info(f"Weighted F1-Score: {accuracy_result['f1']:.4f}")
        logger.info(f"Classification Report:\n{accuracy_result['report']}")
    logger.info("-" * 50)

    logger.info("\n" + "=" * 20 + " Comparison Complete " + "=" * 20)


if __name__ == '__main__':
    main()