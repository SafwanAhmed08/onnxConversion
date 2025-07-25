import torch
import torch.nn as nn
from utils.parse_config import parse_model_config
from utils.build_modules import create_modules
import numpy as np

class Darknet(nn.Module):
    def __init__(self, cfg_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(cfg_path)
        self.module_list, self.yolo_layers = create_modules(self.module_defs)
        self.img_size = img_size

    def load_weights(self, weights_path):
        with open(weights_path, 'rb') as f:
            header = torch.from_numpy(
                np.fromfile(f, dtype=np.int32, count=5)
            )
            self.header = header
            self.seen = header[3]
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] != 'convolutional':
                continue
            conv = module[0]
            if 'batch_normalize' in module_def:
                bn = module[1]
                num_b = bn.bias.numel()
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b])); ptr += num_b
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b])); ptr += num_b
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b])); ptr += num_b
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b])); ptr += num_b
            else:
                num_b = conv.bias.numel()
                conv.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b])); ptr += num_b
            num_w = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_w]).view_as(conv.weight)); ptr += num_w

    def forward(self, x):
        outputs = []
        layer_outputs = []

        for i, module in enumerate(self.module_list):
            mtype = self.module_defs[i]['type']

            if mtype in ['convolutional', 'upsample']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in self.module_defs[i]['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    x = torch.cat([layer_outputs[layer] for layer in layers], 1)
            elif mtype == 'shortcut':
                from_layer = int(self.module_defs[i]['from'])
                x = x + layer_outputs[from_layer]
            elif mtype == 'yolo':
                outputs.append(x)
            layer_outputs.append(x)

        return tuple(outputs)  # (output0, output1, output2)
