"""TensorRT Inference Engine

使用TensorRT进行高性能GPU推理
支持FP32, FP16, INT8量化
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("[WARNING] TensorRT not installed")


class TensorRTPredictor:
    """TensorRT推理器"""

    def __init__(self, engine_path):
        """
        Args:
            engine_path: TensorRT engine文件路径
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not installed")

        # 加载engine
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # 分配GPU内存
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # 分配host和device内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
                input_shape = self.engine.get_binding_shape(binding)
                self.input_size = (input_shape[2], input_shape[3])
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

        print(f"[INFO] TensorRT engine loaded: {engine_path}")
        print(f"[INFO] Input shape: {input_shape}")

    def preprocess(self, image_path):
        """预处理"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.input_size, Image.BILINEAR)

        img_array = np.array(img).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array.ravel()

    def predict(self, image_path, top_k=5):
        """预测"""
        # 预处理
        input_data = self.preprocess(image_path)

        # 拷贝到GPU
        np.copyto(self.inputs[0]['host'], input_data)

        # 推理
        start_time = time.time()

        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)

        self.stream.synchronize()

        inference_time = time.time() - start_time

        # 获取输出
        logits = self.outputs[0]['host']

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Top-K
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]

        return {
            'top_k': [(int(idx), float(prob)) for idx, prob in zip(top_indices, top_probs)],
            'inference_time': inference_time
        }

    def benchmark(self, image_path, num_runs=100):
        """性能测试"""
        print(f"\n[INFO] Running TensorRT benchmark ({num_runs} iterations)...")

        input_data = self.preprocess(image_path)
        np.copyto(self.inputs[0]['host'], input_data)

        # Warmup
        for _ in range(10):
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()
            times.append(time.time() - start)

        times = np.array(times) * 1000

        print(f"\n{'='*80}")
        print(f"{'TensorRT Benchmark Results':^80}")
        print(f"{'='*80}")
        print(f"Runs: {num_runs}")
        print(f"Mean: {times.mean():.2f} ms")
        print(f"Std: {times.std():.2f} ms")
        print(f"Min: {times.min():.2f} ms")
        print(f"Max: {times.max():.2f} ms")
        print(f"FPS: {1000/times.mean():.2f}")
        print(f"{'='*80}\n")


def build_engine_from_onnx(onnx_path, engine_path, fp16=False, int8=False, workspace_size=1<<30):
    """从ONNX构建TensorRT引擎"""
    if not TRT_AVAILABLE:
        raise ImportError("TensorRT not installed")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # 解析ONNX
    print(f"[INFO] Parsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX")

    # 配置
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] FP16 mode enabled")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("[INFO] INT8 mode enabled (requires calibration)")

    # 构建引擎
    print("[INFO] Building TensorRT engine (this may take a while)...")
    engine = builder.build_engine(network, config)

    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"[INFO] TensorRT engine saved to: {engine_path}")


def parse_args():
    p = argparse.ArgumentParser()

    # Build or Predict
    p.add_argument('--mode', choices=['build', 'predict', 'benchmark'], default='predict')

    # Build mode
    p.add_argument('--onnx', help='ONNX model path (for build mode)')
    p.add_argument('--engine', help='TensorRT engine path')
    p.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    p.add_argument('--int8', action='store_true', help='Enable INT8 precision')
    p.add_argument('--workspace', type=int, default=1, help='Workspace size in GB')

    # Predict mode
    p.add_argument('--image', help='Input image path')
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--class_names', help='Class names file')
    p.add_argument('--num_runs', type=int, default=100)

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == 'build':
        if not args.onnx or not args.engine:
            print("Error: --onnx and --engine required for build mode")
            return

        workspace_size = args.workspace * (1 << 30)  # Convert GB to bytes
        build_engine_from_onnx(args.onnx, args.engine, args.fp16, args.int8, workspace_size)

    elif args.mode in ['predict', 'benchmark']:
        if not args.engine or not args.image:
            print("Error: --engine and --image required for predict/benchmark mode")
            return

        # 加载类别名
        class_names = None
        if args.class_names and os.path.exists(args.class_names):
            with open(args.class_names) as f:
                class_names = [line.strip() for line in f]

        # 创建推理器
        predictor = TensorRTPredictor(args.engine)

        if args.mode == 'predict':
            # 预测
            results = predictor.predict(args.image, top_k=args.top_k)

            print(f"\n{'='*80}")
            print(f"{'TensorRT Prediction Results':^80}")
            print(f"{'='*80}")
            print(f"Image: {args.image}")
            print(f"Inference Time: {results['inference_time']*1000:.2f} ms")
            print(f"\nTop-{args.top_k} Predictions:")

            for i, (cls_id, prob) in enumerate(results['top_k']):
                cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"class_{cls_id}"
                print(f"  {i+1}. {cls_name}: {prob*100:.2f}%")

            print(f"{'='*80}\n")

        else:  # benchmark
            predictor.benchmark(args.image, num_runs=args.num_runs)


if __name__ == "__main__":
    main()

