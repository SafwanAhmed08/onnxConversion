# utils/parse_config.py

def parse_model_config(path):
    """Parses the YOLO config file and returns module definitions."""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    module_defs = []
    for line in lines:
        if line.startswith('['):  # Start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            module_defs[-1][key.rstrip()] = value.lstrip()
    return module_defs
