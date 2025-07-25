import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import Darknet

cfg = './cfg/yolov3.cfg'
pth = './weights/yolov3.pth'
onnx = './weights/yolov3.onnx'

model = Darknet(cfg)
model.load_state_dict(torch.load(pth, map_location='cpu'))
model.eval()

dummy_input = torch.randn(1, 3, 416, 416)

torch.onnx.export(
    model,
    dummy_input,
    onnx,
    verbose=True,
    input_names=['input'],
    output_names=['output0', 'output1', 'output2'],
    opset_version=11
)

print(f"Exported to ONNX: {onnx}")
