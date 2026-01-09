"""Face Recognition Evaluation and Inference

人脸识别评估和推理工具
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ptcls.arch.backbone import build_backbone
from ptcls.face import FaceVerification, FaceIdentification, FaceQualityAssessment


def parse_args():
    p = argparse.ArgumentParser()

    # Task
    p.add_argument('--task', required=True, choices=['verify', 'identify', 'build_gallery', 'quality'])

    # Model
    p.add_argument('--model', default='resnet50')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--embedding_size', type=int, default=512)
    p.add_argument('--device', default='cuda')

    # Verify task
    p.add_argument('--image1', help='First image for verification')
    p.add_argument('--image2', help='Second image for verification')
    p.add_argument('--threshold', type=float, default=0.3, help='Similarity threshold')

    # Identify task
    p.add_argument('--query', help='Query image for identification')
    p.add_argument('--gallery_dir', help='Gallery directory')
    p.add_argument('--gallery_file', help='Gallery features file')
    p.add_argument('--top_k', type=int, default=5)

    # Quality assessment
    p.add_argument('--image', help='Image for quality assessment')

    return p.parse_args()


def load_model(args):
    """加载模型"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 加载backbone
    model = build_backbone(args.model, num_classes=args.embedding_size)

    # 加载权重
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()

    return model, device


def load_image(image_path):
    """加载和预处理图片"""
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img = Image.open(image_path).convert('RGB')
    return transform(img)


def task_verify(args):
    """1:1 人脸验证"""
    if not args.image1 or not args.image2:
        print("Error: --image1 and --image2 required for verify task")
        return

    # 加载模型
    model, device = load_model(args)

    # 创建验证器
    verifier = FaceVerification(model, threshold=args.threshold, device=device)

    # 加载图片
    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    # 验证
    is_same, similarity = verifier.verify(img1, img2)

    print(f"\n{'='*80}")
    print(f"{'Face Verification Result':^80}")
    print(f"{'='*80}")
    print(f"Image 1: {args.image1}")
    print(f"Image 2: {args.image2}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Threshold: {args.threshold:.4f}")
    print(f"Result: {'✓ SAME PERSON' if is_same else '✗ DIFFERENT PERSONS'}")
    print(f"{'='*80}\n")


def task_identify(args):
    """1:N 人脸识别"""
    if not args.query:
        print("Error: --query required for identify task")
        return

    # 加载模型
    model, device = load_model(args)

    # 创建识别器
    identifier = FaceIdentification(model, threshold=args.threshold, device=device)

    # 构建人脸库
    if args.gallery_dir:
        print("[INFO] Building face gallery...")
        gallery_dir = Path(args.gallery_dir)

        images = []
        labels = []
        person_names = []

        person_dirs = sorted([d for d in gallery_dir.iterdir() if d.is_dir()])

        for person_idx, person_dir in enumerate(person_dirs):
            person_name = person_dir.name
            person_names.append(person_name)

            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in person_dir.glob(ext):
                    images.append(load_image(str(img_path)))
                    labels.append(person_idx)

        identifier.build_gallery(images, labels)

    elif args.gallery_file:
        # 从文件加载预构建的人脸库
        import pickle
        with open(args.gallery_file, 'rb') as f:
            gallery_data = pickle.load(f)
        identifier.gallery_features = gallery_data['features']
        identifier.gallery_labels = gallery_data['labels']
        person_names = gallery_data.get('names', [])

    else:
        print("Error: --gallery_dir or --gallery_file required for identify task")
        return

    # 加载查询图片
    query_img = load_image(args.query)

    # 识别
    results = identifier.identify(query_img, top_k=args.top_k)

    print(f"\n{'='*80}")
    print(f"{'Face Identification Result':^80}")
    print(f"{'='*80}")
    print(f"Query: {args.query}")
    print(f"\nTop-{len(results)} Matches:")

    if results:
        for i, (label, similarity) in enumerate(results):
            person_name = person_names[label] if label < len(person_names) else f"person_{label}"
            print(f"  {i+1}. {person_name} (similarity: {similarity:.4f})")
    else:
        print("  No match found (all similarities below threshold)")

    print(f"{'='*80}\n")


def task_build_gallery(args):
    """构建人脸库并保存"""
    if not args.gallery_dir:
        print("Error: --gallery_dir required for build_gallery task")
        return

    # 加载模型
    model, device = load_model(args)

    # 创建识别器
    identifier = FaceIdentification(model, device=device)

    # 构建人脸库
    print("[INFO] Building face gallery...")
    gallery_dir = Path(args.gallery_dir)

    images = []
    labels = []
    person_names = []

    person_dirs = sorted([d for d in gallery_dir.iterdir() if d.is_dir()])

    for person_idx, person_dir in enumerate(person_dirs):
        person_name = person_dir.name
        person_names.append(person_name)

        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in person_dir.glob(ext):
                images.append(load_image(str(img_path)))
                labels.append(person_idx)

        print(f"  {person_name}: {sum(1 for l in labels if l == person_idx)} images")

    identifier.build_gallery(images, labels)

    # 保存
    save_path = args.gallery_file or 'face_gallery.pkl'
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump({
            'features': identifier.gallery_features,
            'labels': identifier.gallery_labels,
            'names': person_names
        }, f)

    print(f"[INFO] Gallery saved to: {save_path}")


def task_quality(args):
    """人脸质量评估"""
    if not args.image:
        print("Error: --image required for quality task")
        return

    # 加载图片
    img = load_image(args.image)

    # 评估质量
    qa = FaceQualityAssessment()
    quality = qa.assess_quality(img)

    print(f"\n{'='*80}")
    print(f"{'Face Quality Assessment':^80}")
    print(f"{'='*80}")
    print(f"Image: {args.image}")
    print(f"\nQuality Metrics:")
    print(f"  Blur Score: {quality['blur']:.2f} (higher is better)")
    print(f"  Brightness: {quality['brightness']:.2f} (50-200 is good)")
    print(f"  Overall Score: {quality['overall']:.2f}/100")
    print(f"\nQuality: {'✓ GOOD' if quality['is_good'] else '✗ POOR'}")
    print(f"{'='*80}\n")


def main():
    args = parse_args()

    if args.task == 'verify':
        task_verify(args)
    elif args.task == 'identify':
        task_identify(args)
    elif args.task == 'build_gallery':
        task_build_gallery(args)
    elif args.task == 'quality':
        task_quality(args)


if __name__ == "__main__":
    main()

