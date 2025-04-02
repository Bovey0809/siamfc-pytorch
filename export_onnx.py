import torch
import torch.nn as nn
import os
from siamfc.siamfc import Net, AlexNetV1, SiamFC

def create_export_model():
    """Create a model for ONNX export with dummy inputs"""
    # Create the model
    model = Net(
        backbone=AlexNetV1(),
        head=SiamFC(out_scale=0.001)
    )
    
    # Load the pretrained weights
    if os.path.exists('pretrained/siamfc_alexnet_e50.pth'):
        state_dict = torch.load('pretrained/siamfc_alexnet_e50.pth')
        # Remove 'module.' prefix from state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        raise FileNotFoundError("Pretrained model not found at pretrained/siamfc_alexnet_e50.pth")
    
    model.eval()
    return model

def export_to_onnx():
    # Create model
    model = create_export_model()
    
    # Create dummy inputs
    # z: template image (127x127x3)
    # x: search image (255x255x3)
    dummy_z = torch.randn(1, 3, 127, 127)
    dummy_x = torch.randn(1, 3, 255, 255)
    
    # Export feature extraction model (backbone)
    class FeatureExtractor(nn.Module):
        def __init__(self, backbone):
            super(FeatureExtractor, self).__init__()
            self.backbone = backbone
        
        def forward(self, x):
            return self.backbone(x)
    
    feature_extractor = FeatureExtractor(model.backbone)
    torch.onnx.export(
        feature_extractor,
        dummy_x,
        "siamfc_backbone.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    
    # Export correlation model (head)
    class CorrelationHead(nn.Module):
        def __init__(self, head):
            super(CorrelationHead, self).__init__()
            self.head = head
        
        def forward(self, z, x):
            return self.head(z, x)
    
    correlation_head = CorrelationHead(model.head)
    # Create dummy features
    dummy_z_feat = torch.randn(1, 256, 6, 6)  # template features
    dummy_x_feat = torch.randn(1, 256, 22, 22)  # search features
    
    torch.onnx.export(
        correlation_head,
        (dummy_z_feat, dummy_x_feat),
        "siamfc_head.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['template_features', 'search_features'],
        output_names=['response'],
        dynamic_axes={
            'template_features': {0: 'batch_size'},
            'search_features': {0: 'batch_size'},
            'response': {0: 'batch_size'}
        }
    )
    
    print("Models exported successfully:")
    print("1. siamfc_backbone.onnx - Feature extraction model")
    print("2. siamfc_head.onnx - Feature correlation model")

if __name__ == '__main__':
    export_to_onnx() 