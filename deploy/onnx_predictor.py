"""ONNX Inference Engine

使用ONNX Runtime进行高性能推理
支持CPU和GPU加速
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[WARNING] onnxruntime not installed. Install: pip install onnxruntime or onnxruntime-gpu")


class ONNXPredictor:
    """ONNX推理器"""

    def __init__(self, model_path, device='cpu'):
        """
        Args:
            model_path: ONNX模型路径
            device: 'cpu' or 'cuda'
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed")

        # 设置providers
        providers = []
        if device == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        # 创建session
        self.session = ort.InferenceSession(model_path, providers=providers)

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (input_shape[2], input_shape[3])

        print(f"[INFO] ONNX model loaded: {model_path}")
        print(f"[INFO] Input: {self.input_name}, shape: {input_shape}")
        print(f"[INFO] Output: {self.output_name}")
        print(f"[INFO] Provider: {self.session.get_providers()}")

    def preprocess(self, image_path):
        """预处理图片"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.input_size, Image.BILINEAR)

        # 转numpy并归一化
        img_array = np.array(img).astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image_path, top_k=5):
        """预测"""
        # 预处理
        input_data = self.preprocess(image_path)

        # 推理
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        inference_time = time.time() - start_time

        # Softmax
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Top-K
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]

        results = {
            'top_k': [(int(idx), float(prob)) for idx, prob in zip(top_indices, top_probs)],
            'inference_time': inference_time
        }

        return results

    def benchmark(self, image_path, num_runs=100):
        """性能测试"""
        print(f"\n[INFO] Running benchmark ({num_runs} iterations)...")

        input_data = self.preprocess(image_path)

        # Warmup
        for _ in range(10):
            self.session.run([self.output_name], {self.input_name: input_data})

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            self.session.run([self.output_name], {self.input_name: input_data})
            times.append(time.time() - start)

        times = np.array(times) * 1000  # Convert to ms

        print(f"\n{'='*80}")
        print(f"{'Benchmark Results':^80}")
        print(f"{'='*80}")
        print(f"Runs: {num_runs}")
        print(f"Mean: {times.mean():.2f} ms")
        print(f"Std: {times.std():.2f} ms")
        print(f"Min: {times.min():.2f} ms")
        print(f"Max: {times.max():.2f} ms")
        print(f"FPS: {1000/times.mean():.2f}")
        print(f"{'='*80}\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='ONNX model path')
    p.add_argument('--image', required=True, help='Input image path')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--benchmark', action='store_true', help='Run benchmark')
    p.add_argument('--num_runs', type=int, default=100)
    p.add_argument('--class_names', help='Class names file (one per line)')
    return p.parse_args()


def main():
    args = parse_args()

    # 加载类别名
    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names) as f:
            class_names = [line.strip() for line in f]

    # 创建推理器
    predictor = ONNXPredictor(args.model, device=args.device)

    # 预测
    results = predictor.predict(args.image, top_k=args.top_k)

    print(f"\n{'='*80}")
    print(f"{'Prediction Results':^80}")
    print(f"{'='*80}")
    print(f"Image: {args.image}")
    print(f"Inference Time: {results['inference_time']*1000:.2f} ms")
    print(f"\nTop-{args.top_k} Predictions:")

    for i, (cls_id, prob) in enumerate(results['top_k']):
        cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"class_{cls_id}"
        print(f"  {i+1}. {cls_name}: {prob*100:.2f}%")

    print(f"{'='*80}\n")

    # Benchmark
    if args.benchmark:
        predictor.benchmark(args.image, num_runs=args.num_runs)


if __name__ == "__main__":
    main()

