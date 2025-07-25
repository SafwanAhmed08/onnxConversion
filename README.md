# YOLOv3 Model Conversion

This project provides utilities to convert YOLOv3 models from Darknet format to PyTorch (`.pth`) and ONNX (`.onnx`) formats.

## Project Structure

```
cfg/           # YOLOv3 configuration files (.cfg)
convert/       # Conversion scripts
models/        # Model definitions
utils/         # Utility functions
weights/       # Model weights (.weights, .pth, .onnx)
```

## Getting Started

1. **Obtain new weights and cfg files**  
   Download the required YOLOv3 `.weights` and `.cfg` files from the [Darknet website](https://pjreddie.com/darknet/yolo/) or your custom `.weights` and `.cfg` and place them in the `weights/` and `cfg/` folders respectively. 

2. **Convert to PyTorch format**  
   Run the following command to convert Darknet weights to PyTorch:
   ```sh
   python ./convert/convert_to_pth.py
   ```

3. **Convert to ONNX format**  
   After generating the `.pth` file, convert it to ONNX format:
   ```sh
   python ./convert/convert_to_onnx.py
   ```

## Files

- `convert/convert_to_pth.py`: Converts Darknet weights to PyTorch format.
- `convert/convert_to_onnx.py`: Converts PyTorch model to ONNX format.
- `models/model.py`: YOLOv3 model definition.
- `utils/`: Helper functions for building modules and parsing configs.

## Notes

- Make sure to update the `.gitignore` to avoid committing large weight files.
- For custom training, replace the weights and cfg files with your own.
