"""HTTP Serving for Model Deployment

使用Flask提供模型HTTP服务
支持：
- 图像分类
- 人脸识别
- 图像检索

API Endpoints:
- POST /classify - 图像分类
- POST /face/verify - 人脸验证
- POST /face/identify - 人脸识别
- POST /retrieve - 图像检索
"""

import argparse
import os
import sys
import io
import base64
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("[ERROR] Flask not installed. Install: pip install flask")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.arch.backbone import build_backbone


# 全局变量
app = Flask(__name__)
model = None
device = None
transform = None
class_names = None


def load_image_from_request():
    """从HTTP请求加载图片"""
    if 'image' in request.files:
        # 从文件上传
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
    elif 'image_base64' in request.json:
        # 从base64
        img_data = base64.b64decode(request.json['image_base64'])
        image = Image.open(io.BytesIO(img_data)).convert('RGB')
    else:
        return None

    return image


@app.route('/classify', methods=['POST'])
def classify():
    """图像分类接口"""
    try:
        # 加载图片
        image = load_image_from_request()
        if image is None:
            return jsonify({'error': 'No image provided'}), 400

        # 预处理
        img_tensor = transform(image).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            outputs = model(img_tensor)
            if isinstance(outputs, dict):
                outputs = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))

            probs = torch.softmax(outputs, dim=1)[0]
            top5_prob, top5_idx = probs.topk(5)

        # 构建响应
        results = []
        for prob, idx in zip(top5_prob, top5_idx):
            cls_name = class_names[idx] if class_names and idx < len(class_names) else f"class_{idx}"
            results.append({
                'class_id': int(idx),
                'class_name': cls_name,
                'probability': float(prob)
            })

        return jsonify({
            'success': True,
            'predictions': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/face/verify', methods=['POST'])
def face_verify():
    """人脸验证接口（1:1）"""
    try:
        # 需要两张图片
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Two images required'}), 400

        img1 = Image.open(request.files['image1'].stream).convert('RGB')
        img2 = Image.open(request.files['image2'].stream).convert('RGB')

        # 特征提取
        feat1 = extract_feature(img1)
        feat2 = extract_feature(img2)

        # 计算相似度
        similarity = float((feat1 * feat2).sum())

        threshold = request.json.get('threshold', 0.3) if request.json else 0.3
        is_same = similarity >= threshold

        return jsonify({
            'success': True,
            'is_same_person': is_same,
            'similarity': similarity,
            'threshold': threshold
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def extract_feature(image):
    """提取特征向量"""
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        if isinstance(outputs, dict):
            feat = outputs.get('embedding', outputs.get('features', list(outputs.values())[0]))
        else:
            feat = outputs

        # L2归一化
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)

    return feat[0]


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/info', methods=['GET'])
def info():
    """模型信息"""
    return jsonify({
        'model_type': 'classification',
        'num_classes': len(class_names) if class_names else 'unknown',
        'device': str(device)
    })


def parse_args():
    p = argparse.ArgumentParser(description="Model HTTP Serving")

    # Model
    p.add_argument('--model', required=True, help='Model name')
    p.add_argument('--checkpoint', required=True, help='Model checkpoint')
    p.add_argument('--num_classes', type=int, default=1000)
    p.add_argument('--class_names', help='Class names file')

    # Server
    p.add_argument('--host', default='0.0.0.0', help='Server host')
    p.add_argument('--port', type=int, default=8080, help='Server port')
    p.add_argument('--device', default='cuda', help='Device')

    return p.parse_args()


def main():
    global model, device, transform, class_names

    if not FLASK_AVAILABLE:
        print("[ERROR] Flask not installed")
        return

    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # 加载模型
    print(f"[INFO] Loading model: {args.model}")
    model = build_backbone(args.model, num_classes=args.num_classes)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    model = model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully")

    # 加载类别名
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names) as f:
            class_names = [line.strip() for line in f]
        print(f"[INFO] Loaded {len(class_names)} class names")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 启动服务器
    print(f"\n{'='*80}")
    print(f"{'Model HTTP Server Starting':^80}")
    print(f"{'='*80}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Model: {args.model}")
    print(f"\nAPI Endpoints:")
    print(f"  POST http://{args.host}:{args.port}/classify")
    print(f"  POST http://{args.host}:{args.port}/face/verify")
    print(f"  GET  http://{args.host}:{args.port}/health")
    print(f"  GET  http://{args.host}:{args.port}/info")
    print(f"{'='*80}\n")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

