import torch
import argparse
from pipeline import Pipeline
from pipelineV7 import PipelineV7
from segmentor import RoadSegmenter

import onnx_graphsurgeon

print("onnx_graphsurgeon version:", onnx_graphsurgeon.__version__)
print(f"Using torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Cuda: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device available'}")

# Using torch version: 2.9.1+cu126, CUDA available: True
# CUDA device count: 1
# Cuda: NVIDIA GeForce RTX 4050 Laptop GPU