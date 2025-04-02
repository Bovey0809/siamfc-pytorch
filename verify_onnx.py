import onnx
import numpy as np
import torch
from siamfc.siamfc import Net, AlexNetV1, SiamFC
import os
import cv2
import onnxruntime

def verify_backbone():
    print("\n=== Verifying Backbone Model ===")
    
    # Load the backbone ONNX model
    backbone_model = onnx.load("siamfc_backbone.onnx")
    
    # Check model structure
    print("\n=== Backbone Model Structure ===")
    print(f"IR version: {backbone_model.ir_version}")
    print(f"Opset version: {backbone_model.opset_import[0].version}")
    
    # Print input/output specifications
    print("\n=== Backbone Input/Output Specifications ===")
    for input in backbone_model.graph.input:
        print(f"Input '{input.name}':")
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_value:
                print(f"  - Dimension: {dim.dim_value}")
            else:
                print(f"  - Dynamic dimension: {dim.dim_param}")
    
    for output in backbone_model.graph.output:
        print(f"Output '{output.name}':")
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_value:
                print(f"  - Dimension: {dim.dim_value}")
            else:
                print(f"  - Dynamic dimension: {dim.dim_param}")
    
    # Create sample inputs
    template = torch.randn(1, 3, 255, 255, dtype=torch.float32)
    search = torch.randn(1, 3, 255, 255, dtype=torch.float32)
    
    # Get PyTorch model output for comparison
    pytorch_model = Net(backbone=AlexNetV1(), head=SiamFC(out_scale=0.001))
    if os.path.exists('pretrained/siamfc_alexnet_e50.pth'):
        state_dict = torch.load('pretrained/siamfc_alexnet_e50.pth')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        pytorch_model.load_state_dict(new_state_dict, strict=False)
    else:
        raise FileNotFoundError("Pretrained model not found at pretrained/siamfc_alexnet_e50.pth")
    
    pytorch_model.eval()
    
    with torch.no_grad():
        pytorch_template_feat = pytorch_model.backbone(template)
        pytorch_search_feat = pytorch_model.backbone(search)
    
    # Load ONNX model and run inference
    session = onnxruntime.InferenceSession("siamfc_backbone.onnx")
    
    # Run inference for template
    onnx_template_feat = session.run(
        ['features'],
        {'input': template.numpy().astype(np.float32)}
    )[0]
    
    # Run inference for search
    onnx_search_feat = session.run(
        ['features'],
        {'input': search.numpy().astype(np.float32)}
    )[0]
    
    # Compare outputs
    print("\n=== Backbone Output Comparison ===")
    print(f"Template features:")
    print(f"  PyTorch shape: {pytorch_template_feat.shape}")
    print(f"  ONNX shape: {onnx_template_feat.shape}")
    print(f"  Max diff: {np.max(np.abs(pytorch_template_feat.numpy() - onnx_template_feat))}")
    
    print(f"\nSearch features:")
    print(f"  PyTorch shape: {pytorch_search_feat.shape}")
    print(f"  ONNX shape: {onnx_search_feat.shape}")
    print(f"  Max diff: {np.max(np.abs(pytorch_search_feat.numpy() - onnx_search_feat))}")
    
    return pytorch_template_feat, pytorch_search_feat, onnx_template_feat, onnx_search_feat

def verify_head(template_feat, search_feat):
    print("\n=== Verifying Head Model ===")
    
    # Load the head ONNX model
    head_model = onnx.load("siamfc_head.onnx")
    
    # Check model structure
    print("\n=== Head Model Structure ===")
    print(f"IR version: {head_model.ir_version}")
    print(f"Opset version: {head_model.opset_import[0].version}")
    
    # Print input/output specifications
    print("\n=== Head Input/Output Specifications ===")
    for input in head_model.graph.input:
        print(f"Input '{input.name}':")
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_value:
                print(f"  - Dimension: {dim.dim_value}")
            else:
                print(f"  - Dynamic dimension: {dim.dim_param}")
    
    for output in head_model.graph.output:
        print(f"Output '{output.name}':")
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_value:
                print(f"  - Dimension: {dim.dim_value}")
            else:
                print(f"  - Dynamic dimension: {dim.dim_param}")
    
    # Get PyTorch model output for comparison
    pytorch_model = Net(backbone=AlexNetV1(), head=SiamFC(out_scale=0.001))
    if os.path.exists('pretrained/siamfc_alexnet_e50.pth'):
        state_dict = torch.load('pretrained/siamfc_alexnet_e50.pth')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        pytorch_model.load_state_dict(new_state_dict, strict=False)
    else:
        raise FileNotFoundError("Pretrained model not found at pretrained/siamfc_alexnet_e50.pth")
    
    pytorch_model.eval()
    
    with torch.no_grad():
        pytorch_output = pytorch_model.head(template_feat, search_feat)
    
    # Load ONNX model and run inference
    session = onnxruntime.InferenceSession("siamfc_head.onnx")
    
    # Run inference
    onnx_output = session.run(
        ['response'],
        {
            'template_features': template_feat.numpy().astype(np.float32),
            'search_features': search_feat.numpy().astype(np.float32)
        }
    )[0]
    
    # Compare outputs
    print("\n=== Head Output Comparison ===")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"Maximum difference: {np.max(np.abs(pytorch_output.numpy() - onnx_output))}")
    
    # Check if outputs are close enough
    is_close = np.allclose(pytorch_output.numpy(), onnx_output, rtol=1e-5, atol=1e-5)
    print(f"Outputs are close enough: {is_close}")

def verify_onnx_model():
    # First verify backbone
    pytorch_template_feat, pytorch_search_feat, onnx_template_feat, onnx_search_feat = verify_backbone()
    
    # Then verify head with the features
    verify_head(pytorch_template_feat, pytorch_search_feat)

if __name__ == '__main__':
    verify_onnx_model() 