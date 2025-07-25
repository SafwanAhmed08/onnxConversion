import torch
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import Darknet

cfg = './cfg/yolov3.cfg'
weights = './weights/yolov3.weights'
output = './weights/yolov3.pth'

model = Darknet(cfg)
model.load_weights(weights)
torch.save(model.state_dict(), output)
print(f"Saved: {output}")
