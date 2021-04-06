"""Script to export a model to onnx."""
import argparse
import os
import torch

from model import registered_models

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to export")
    parser.add_argument("--output", type=str, required=True, help="Output dir.")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        raise ValueError(f"Output path {args.output} doesn't exist")

    model_builder = registered_models[args.model]
    model = model_builder(100, True).eval()

    onnx_path = os.path.join(args.output, "model-"+model.name()+".onnx")
    mocked_input = torch.zeros([1] + model.input_shape(), dtype=torch.float32) 
    torch.onnx.export(
        model=model,
        args=mocked_input,
        f=onnx_path,
        export_params=True
    )
    print(f"Done with exporting {onnx_path}")
