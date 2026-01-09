"""Setup script for visionhub"""
from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='visionhub',
    version='1.0.0',
    description='Professional Visual Intelligence Toolkit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='JKDCPPZzz',
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'numpy>=1.19.0',
        'Pillow>=8.0.0',
        'scipy>=1.5.0',
        'scikit-learn>=0.24.0',
        'tqdm>=4.60.0',
        'pyyaml>=5.4.0',
        'opencv-python>=4.5.0',
        'faiss-cpu>=1.7.0',
        'ultralytics',
    ],
    extras_require={
        'gpu': ['faiss-gpu>=1.7.0'],
        'onnx': ['onnx>=1.10.0', 'onnxruntime>=1.10.0', 'onnx-simplifier>=0.3.0'],
        'serving': ['flask>=2.0.0'],
        'all': ['faiss-gpu>=1.7.0', 'onnx>=1.10.0', 'onnxruntime-gpu>=1.10.0', 'flask>=2.0.0'],
    },
    entry_points={
        'console_scripts': [
            'visionhub-train=visionhub.ptcls.tools.train_classification:main',
            'visionhub-eval=visionhub.ptcls.tools.eval_classification:main',
            'visionhub-export=visionhub.ptcls.tools.export_model:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)

