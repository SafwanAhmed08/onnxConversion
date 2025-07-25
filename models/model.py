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

        # Debug: Print total modules and their types
        print(f"Total modules in module_list: {len(self.module_list)}")
        print(f"Total module_defs: {len(self.module_defs)}")
        for idx, module in enumerate(self.module_list):
            print(f"module_list[{idx}] has {len(list(module.children()))} children")
            for name, layer in module.named_children():
                print(f"  {name}: {type(layer).__name__}")

        ptr = 0
        module_idx = 0
        for i, module_def in enumerate(self.module_defs):
            if module_def['type'] == 'net':
                continue
            if module_def['type'] != 'convolutional':
                module_idx += 1
                continue
            
            print(f"Processing convolutional module_def {i}, module_idx {module_idx}")
            module = self.module_list[module_idx]
            
            # Access by name instead of index
            conv = None
            bn = None
            for name, layer in module.named_children():
                if isinstance(layer, nn.Conv2d):
                    conv = layer
                elif isinstance(layer, nn.BatchNorm2d):
                    bn = layer
            
            if conv is None:
                raise ValueError(f"No Conv2d layer found in module {module_idx}")
            
            if 'batch_normalize' in module_def and bn is not None:
                num_b = bn.bias.numel()
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b])); ptr += num_b
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b])); ptr += num_b
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b])); ptr += num_b
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b])); ptr += num_b
            else:
                if conv.bias is not None:
                    num_b = conv.bias.numel()
                    conv.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b])); ptr += num_b
            num_w = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_w]).view_as(conv.weight)); ptr += num_w
            module_idx += 1

    def forward(self, x):
        outputs = []
        layer_outputs = []

        module_idx = 0
        for i, module_def in enumerate(self.module_defs):
            if module_def['type'] == 'net':
                continue
                
            module = self.module_list[module_idx]
            mtype = module_def['type']

            if mtype in ['convolutional', 'upsample']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in module_def['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    x = torch.cat([layer_outputs[layer] for layer in layers], 1)
            elif mtype == 'shortcut':
                from_layer = int(module_def['from'])
                x = x + layer_outputs[from_layer]
            elif mtype == 'yolo':
                outputs.append(x)
            layer_outputs.append(x)
            module_idx += 1

        return tuple(outputs)  # (output0, output1, output2)
