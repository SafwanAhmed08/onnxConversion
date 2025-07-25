import torch.nn as nn

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

def create_modules(module_defs):
    module_list = nn.ModuleList()
    yolo_layers = []
    output_filters = [3]  # Assume RGB input
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            pad = (kernel_size - 1) // 2 if int(module_def.get('pad', 1)) else 0
            activation = module_def.get('activation', None)

            conv = nn.Conv2d(
                in_channels=output_filters[-1],
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=not int(module_def.get('batch_normalize', 0))
            )
            modules.add_module(f"conv_{i}", conv)

            if int(module_def.get('batch_normalize', 0)):
                modules.add_module(f"batch_norm_{i}", nn.BatchNorm2d(filters))

            if activation == 'leaky':
                modules.add_module(f"leaky_{i}", nn.LeakyReLU(0.1))

            output_filters.append(filters)

        elif module_def['type'] == 'shortcut':
            modules.add_module(f"shortcut_{i}", EmptyLayer())
            output_filters.append(output_filters[-1])

        elif module_def['type'] == 'route':
            modules.add_module(f"route_{i}", EmptyLayer())
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[layer] if layer > 0 else output_filters[layer] for layer in layers])
            output_filters.append(filters)

        elif module_def['type'] == 'upsample':
            modules.add_module(f"upsample_{i}", nn.Upsample(scale_factor=2, mode='nearest'))
            output_filters.append(output_filters[-1])

        elif module_def['type'] == 'yolo':
            yolo_layers.append(i)
            modules.add_module(f"yolo_{i}", EmptyLayer())
            output_filters.append(output_filters[-1])

        module_list.append(modules)

    return module_list, yolo_layers
