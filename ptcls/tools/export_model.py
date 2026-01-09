"""Model Export Tool

将PyTorch模型导出为多种部署格式：
- ONNX (跨平台标准)
- TorchScript (PyTorch原生)
- OpenVINO (Intel优化)
- TensorRT (NVIDIA优化) - 需要安装torch2trt
- CoreML (Apple设备) - 需要安装coremltools

支持：
- 分类模型
- 检索模型
- 人脸识别模型
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.arch.backbone import build_backbone


def parse_args():
    p = argparse.ArgumentParser(description="Export PyTorch Model to Deployment Formats")

    # Model
    p.add_argument('--model', required=True, help='Model name (resnet50, efficientnet_b0, etc.)')
    p.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    p.add_argument('--num_classes', type=int, default=1000, help='Number of classes')

    # Export format
    p.add_argument('--format', required=True,
                   choices=['onnx', 'torchscript', 'openvino', 'tensorrt', 'coreml'],
                   help='Export format')

    # Input shape
    p.add_argument('--input_size', type=int, nargs='+', default=[224, 224],
                   help='Input size (H W)')
    p.add_argument('--batch_size', type=int, default=1, help='Batch size for export')

    # ONNX specific
    p.add_argument('--opset_version', type=int, default=11, help='ONNX opset version')
    p.add_argument('--simplify', action='store_true', help='Simplify ONNX model')

    # Output
    p.add_argument('--output', default=None, help='Output file path')
    p.add_argument('--save_dir', default='deploy/exported_models', help='Save directory')

    # Optimization
    p.add_argument('--half', action='store_true', help='Export FP16 model')
    p.add_argument('--dynamic', action='store_true', help='Dynamic batch size (ONNX)')

    return p.parse_args()


def load_model(args):
    """加载PyTorch模型"""
    print(f"[INFO] Loading model: {args.model}")

    model = build_backbone(args.model, num_classes=args.num_classes)

    # 加载权重
    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    model.eval()

    # FP16
    if args.half:
        model = model.half()
        print("[INFO] Model converted to FP16")

    return model


def export_onnx(model, args, save_path):
    """导出ONNX格式"""
    print(f"\n[INFO] Exporting to ONNX...")

    # 准备输入
    h, w = args.input_size if len(args.input_size) == 2 else (args.input_size[0], args.input_size[0])
    dummy_input = torch.randn(args.batch_size, 3, h, w)

    if args.half:
        dummy_input = dummy_input.half()

    # 动态轴
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    print(f"[INFO] ONNX model saved to: {save_path}")

    # 简化（需要安装 onnx-simplifier）
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify

            print("[INFO] Simplifying ONNX model...")
            onnx_model = onnx.load(save_path)
            onnx_model_sim, check = simplify(onnx_model)

            if check:
                onnx.save(onnx_model_sim, save_path)
                print("[INFO] ONNX model simplified successfully")
            else:
                print("[WARNING] ONNX simplification failed")
        except ImportError:
            print("[WARNING] onnx-simplifier not installed, skipping simplification")
            print("          Install: pip install onnx-simplifier")

    # 验证
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("[INFO] ONNX model validation passed")
    except Exception as e:
        print(f"[WARNING] ONNX validation failed: {e}")


def export_torchscript(model, args, save_path):
    """导出TorchScript格式"""
    print(f"\n[INFO] Exporting to TorchScript...")

    h, w = args.input_size if len(args.input_size) == 2 else (args.input_size[0], args.input_size[0])
    dummy_input = torch.randn(args.batch_size, 3, h, w)

    if args.half:
        dummy_input = dummy_input.half()

    # Trace模式
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(save_path)
        print(f"[INFO] TorchScript model saved to: {save_path}")

        # 验证
        output_orig = model(dummy_input)
        output_script = traced_model(dummy_input)

        if isinstance(output_orig, dict):
            output_orig = output_orig.get('logits', output_orig.get('output', list(output_orig.values())[0]))
        if isinstance(output_script, dict):
            output_script = output_script.get('logits', output_script.get('output', list(output_script.values())[0]))

        diff = torch.abs(output_orig - output_script).max().item()
        print(f"[INFO] TorchScript validation: max diff = {diff:.6f}")

    except Exception as e:
        print(f"[ERROR] TorchScript export failed: {e}")


def export_openvino(model, args, save_path):
    """导出OpenVINO格式"""
    print(f"\n[INFO] Exporting to OpenVINO...")

    try:
        # 首先导出ONNX
        onnx_path = save_path.replace('.xml', '.onnx')
        export_onnx(model, args, onnx_path)

        # 使用OpenVINO转换工具
        print("[INFO] Converting ONNX to OpenVINO IR...")
        print("[INFO] Please use OpenVINO Model Optimizer:")
        print(f"       mo --input_model {onnx_path} --output_dir {os.path.dirname(save_path)}")

    except Exception as e:
        print(f"[ERROR] OpenVINO export failed: {e}")


def export_tensorrt(model, args, save_path):
    """导出TensorRT格式"""
    print(f"\n[INFO] Exporting to TensorRT...")

    try:
        from torch2trt import torch2trt

        h, w = args.input_size if len(args.input_size) == 2 else (args.input_size[0], args.input_size[0])
        dummy_input = torch.randn(args.batch_size, 3, h, w).cuda()
        model = model.cuda()

        if args.half:
            dummy_input = dummy_input.half()
            model = model.half()

        # 转换
        model_trt = torch2trt(
            model,
            [dummy_input],
            fp16_mode=args.half,
            max_batch_size=args.batch_size
        )

        torch.save(model_trt.state_dict(), save_path)
        print(f"[INFO] TensorRT model saved to: {save_path}")

    except ImportError:
        print("[ERROR] torch2trt not installed")
        print("        Install: pip install torch2trt")
    except Exception as e:
        print(f"[ERROR] TensorRT export failed: {e}")


def export_coreml(model, args, save_path):
    """导出CoreML格式"""
    print(f"\n[INFO] Exporting to CoreML...")

    try:
        import coremltools as ct

        h, w = args.input_size if len(args.input_size) == 2 else (args.input_size[0], args.input_size[0])
        dummy_input = torch.randn(args.batch_size, 3, h, w)

        # Trace模型
        traced_model = torch.jit.trace(model, dummy_input)

        # 转换为CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(shape=(args.batch_size, 3, h, w), name="input")],
        )

        coreml_model.save(save_path)
        print(f"[INFO] CoreML model saved to: {save_path}")

    except ImportError:
        print("[ERROR] coremltools not installed")
        print("        Install: pip install coremltools")
    except Exception as e:
        print(f"[ERROR] CoreML export failed: {e}")


def main():
    args = parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载模型
    model = load_model(args)

    # 确定输出路径
    if args.output:
        save_path = args.output
    else:
        model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        ext_map = {
            'onnx': '.onnx',
            'torchscript': '.pt',
            'openvino': '.xml',
            'tensorrt': '_trt.pth',
            'coreml': '.mlmodel'
        }
        ext = ext_map.get(args.format, '.bin')
        save_path = os.path.join(args.save_dir, f"{model_name}{ext}")

    print(f"\n{'='*80}")
    print(f"{'Model Export':^80}")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Format: {args.format.upper()}")
    print(f"Input Size: {args.input_size}")
    print(f"Output: {save_path}")
    print(f"{'='*80}\n")

    # 导出
    if args.format == 'onnx':
        export_onnx(model, args, save_path)
    elif args.format == 'torchscript':
        export_torchscript(model, args, save_path)
    elif args.format == 'openvino':
        export_openvino(model, args, save_path)
    elif args.format == 'tensorrt':
        export_tensorrt(model, args, save_path)
    elif args.format == 'coreml':
        export_coreml(model, args, save_path)

    print(f"\n{'='*80}")
    print(f"{'Export Complete!':^80}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

