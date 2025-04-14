import argparse
import copy
import os
import warnings

# Suppress specific warnings if needed, e.g., from Core ML or ONNX
# warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torchvision
from timm import create_model

# Conditional imports based on platform
try:
    import coremltools
except ImportError:
    coremltools = None
    print("Warning: coremltools not found. iOS export will not be available.")

try:
    import onnx
    import tensorflow as tf
except ImportError:
    onnx = None
    tf = None
    print("Warning: onnx or tensorflow not found. Android export will not be available.")

# Assuming your SHViT model definitions are in 'model.py'
# If SHViT is registered with timm correctly, this might not be needed.
# import model
from utilities import utils # Assuming replace_batchnorm is here

def parse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--variant", type=str, required=True, help="Provide shvit model variant name."
    )
    parser.add_argument(
        "--platform",
        type=str,
        required=True,
        choices=['ios', 'android'],
        help="Target platform: 'ios' (Core ML) or 'android' (TFLite)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Provide location to save exported models.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Provide location of trained checkpoint.",
    )
    parser.add_argument(
        "--res_h",
        type=int,
        default=256,
        help="Provide height resolution of input.",
    )
    parser.add_argument(
        "--res_w",
        type=int,
        default=256,
        help="Provide width resolution of input.",
    )
    # Add optional TFLite quantization later if needed
    # parser.add_argument(
    #     "--quantize",
    #     type=str,
    #     default="none",
    #     choices=['none', 'fp16', 'int8'],
    #     help="TFLite quantization mode (android only)."
    # )
    return parser


def export(
    variant: str,
    platform: str,
    output_dir: str,
    checkpoint: str = None,
    res_h: int = 256,
    res_w: int = 256,
    # quantize: str = "none", # Add later
) -> None:
    """Exports the model to Core ML (iOS) or TFLite (Android).

    Args:
        variant: shvit model variant.
        platform: Target platform ('ios' or 'android').
        output_dir: Path to save exported model.
        checkpoint: Path to trained checkpoint. Default: ``None``
        res_h: Input height.
        res_w: Input width.
        # quantize: TFLite quantization mode. Default: ``"none"``
    """
    # --- Common Setup ---
    os.makedirs(output_dir, exist_ok=True)

    # Input tensor shape and dummy data
    input_shape = (1, 3, res_h, res_w)
    dummy_input = torch.randn(input_shape, requires_grad=False)

    # Instantiate model variant using timm
    pytorch_model = create_model(variant, pretrained=False) # Start with no pretrained weights

    # Load checkpoint if provided
    if checkpoint is not None:
        print(f"Loading checkpoint: {checkpoint}")
        # Map location based on available device
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        chkpt = torch.load(checkpoint, map_location=map_location)
        # Adjust key if needed (e.g., if saved with 'model' prefix or 'state_dict')
        state_dict = chkpt.get('state_dict', chkpt.get('model', chkpt))
        if not state_dict:
            raise ValueError(f"Could not find state_dict or model key in checkpoint: {checkpoint}")
        
        # Handle potential DataParallel prefix 'module.'
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            
        # Load the state dict
        pytorch_model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    else:
        print("Warning: No checkpoint provided. Exporting model with initial weights.")


    # Replace BatchNorm layers (often beneficial for inference/export)
    print("Replacing BatchNorm layers...")
    utils.replace_batchnorm(pytorch_model) # Make sure this modifies the model in-place

    print(f"Preparing model {variant} for export...")
    n_parameters = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_parameters / 1e6:.2f} M")

    pytorch_model.eval() # Set model to evaluation mode

    # Define base output path
    output_basename = os.path.join(output_dir, f"{variant}_{res_h}x{res_w}")

    # --- Platform Specific Export ---

    if platform == 'ios':
        if coremltools is None:
            raise ImportError("coremltools is not installed. Cannot export for iOS.")
        print(f"Exporting Core ML model to {output_basename}.mlpackage")

        # 1. Trace the model (Core ML often works well with traced models)
        try:
            traced_model = torch.jit.trace(pytorch_model, dummy_input)
            # Save intermediate traced model (optional, but coremltools.convert often takes file path)
            pt_name = output_basename + "_traced.pt"
            traced_model.save(pt_name)
        except Exception as e:
            print(f"Error during PyTorch JIT tracing: {e}")
            raise

        # 2. Define Core ML input type
        inputs_coreml = [
            coremltools.TensorType(
                name="images", # Input name expected by Core ML model
                shape=dummy_input.shape, # Use the actual shape
            )
        ]

        # 3. Convert to Core ML ML Program format
        try:
            # Convert using the *traced model file path*
            ml_model = coremltools.convert(
                model=pt_name,
                outputs=None, # Automatically infer outputs
                inputs=inputs_coreml,
                convert_to="mlprogram", # Recommended format
                compute_units=coremltools.ComputeUnit.ALL, # Use CPU, GPU, NPU
                # minimum_deployment_target=coremltools.target.iOS15, # Example target
                debug=False,
            )

            # 4. Save the Core ML package
            ml_model.save(output_basename + ".mlpackage")
            print(f"Core ML model saved successfully to {output_basename}.mlpackage")

            # Clean up intermediate traced model
            if os.path.exists(pt_name):
                os.remove(pt_name)

        except Exception as e:
            print(f"Error during Core ML conversion: {e}")
            # Clean up intermediate traced model even if conversion fails
            if os.path.exists(pt_name):
                os.remove(pt_name)
            raise

    elif platform == 'android':
        if onnx is None or tf is None:
            raise ImportError("onnx and/or tensorflow are not installed. Cannot export for Android.")
        print(f"Exporting TFLite model to {output_basename}.tflite")

        # Define input/output names for ONNX
        input_names = ["images"] # Match Core ML input name for consistency
        # Try to get output names dynamically, otherwise use a default
        try:
             # Forward pass to get output structure (optional, can use default)
             outputs = pytorch_model(dummy_input)
             if isinstance(outputs, torch.Tensor):
                 output_names = ["output_0"]
             elif isinstance(outputs, (list, tuple)):
                 output_names = [f"output_{i}" for i in range(len(outputs))]
             else:
                 output_names = ["output_0"] # Fallback
        except Exception:
             output_names = ["output_0"] # Fallback
             print("Warning: Could not dynamically determine output names, using default ['output_0']")


        onnx_path = output_basename + ".onnx"

        # 1. Export to ONNX
        try:
            print(f"Exporting to ONNX: {onnx_path}")
            torch.onnx.export(
                pytorch_model,        # model being run
                dummy_input,          # model input (or a tuple for multiple inputs)
                onnx_path,            # where to save the model
                export_params=True,   # store the trained parameter weights inside the model file
                opset_version=13,     # the ONNX version to export the model to (13 or higher often good)
                do_constant_folding=True, # whether to execute constant folding for optimization
                input_names=input_names,  # the model's input names
                output_names=output_names,# the model's output names
                dynamic_axes={ # Optional: If you want batch size flexibility
                     'images': {0: 'batch_size'},
                     output_names[0]: {0: 'batch_size'}
                 } if len(output_names)==1 else None # Example for single output
            )
            print("ONNX export successful.")
            # Verify ONNX model (optional)
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model checked successfully.")

        except Exception as e:
            print(f"Error during ONNX export: {e}")
            raise

        # 2. Convert ONNX to TFLite using TensorFlow
        try:
            print("Converting ONNX model to TFLite...")
            # Use TensorFlow's converter
            # Note: Direct ONNX conversion might be experimental or require specific TF versions/ops.
            # This attempts conversion; SavedModel might be a more robust intermediate step if this fails.
            converter = tf.lite.TFLiteConverter.from_onnx_model(onnx_path) # Preferred if available & works

            # --- Optional: Add Quantization ---
            # if quantize == 'fp16':
            #     print("Applying FP16 quantization...")
            #     converter.optimizations = [tf.lite.Optimize.DEFAULT]
            #     converter.target_spec.supported_types = [tf.float16]
            # elif quantize == 'int8':
            #     print("Applying INT8 dynamic range quantization...")
            #     converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # # Full INT8 quantization requires a representative dataset:
            # # def representative_dataset():
            # #   for _ in range(100):
            # #     data = ... # Load representative data batch
            # #     yield [tf.cast(data, tf.float32)]
            # # converter.representative_dataset = representative_dataset
            # # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # # converter.inference_input_type = tf.int8 # or tf.uint8
            # # converter.inference_output_type = tf.int8 # or tf.uint8

            # Convert the model
            tflite_model = converter.convert()

            # 3. Save the TFLite model
            tflite_path = output_basename + ".tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print(f"TFLite model saved successfully to {tflite_path}")

        except AttributeError:
             print("Warning: tf.lite.TFLiteConverter.from_onnx_model not found.")
             print("Falling back to converting via SavedModel (requires onnx-tf installed or manual conversion).")
             # Placeholder: Implement ONNX -> SavedModel -> TFLite here if needed
             # Example using onnx_tf (conceptual, needs onnx_tf installed):
             # import onnx_tf.backend as backend
             # onnx_model = onnx.load(onnx_path)
             # tf_rep = backend.prepare(onnx_model)
             # saved_model_dir = output_basename + "_saved_model"
             # tf_rep.export_graph(saved_model_dir)
             # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
             # # ... apply quantization if needed ...
             # tflite_model = converter.convert()
             # tflite_path = output_basename + ".tflite"
             # with open(tflite_path, 'wb') as f:
             #     f.write(tflite_model)
             # print(f"TFLite model saved successfully via SavedModel to {tflite_path}")
             # import shutil
             # shutil.rmtree(saved_model_dir) # Clean up
             print("Error: ONNX->SavedModel->TFLite conversion not fully implemented in this fallback.")
             raise NotImplementedError("ONNX->SavedModel->TFLite conversion required but not implemented.")

        except Exception as e:
            print(f"Error during TFLite conversion: {e}")
            raise
        finally:
             # Clean up intermediate ONNX model
             if os.path.exists(onnx_path):
                 os.remove(onnx_path)

    else:
        # Should not happen due to argparse choices
        print(f"Error: Unsupported platform '{platform}'")
        return

    print("Export process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to export SHViT model to Core ML or TFLite")
    parser = parse_args(parser)
    args = parser.parse_args()

    try:
        export(
            variant=args.variant,
            platform=args.platform,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            res_h=args.res_h,
            res_w=args.res_w,
            # quantize=args.quantize, # Add later
        )
    except Exception as e:
        print(f"An error occurred during export: {e}")
        exit(1)
