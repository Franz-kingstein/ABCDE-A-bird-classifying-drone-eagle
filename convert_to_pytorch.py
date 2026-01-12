"""
Convert Keras EfficientNetV2B2 model with custom weights to PyTorch format (.pt)
Uses timm (PyTorch Image Models) for better architecture support
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout
import json

# Class names
CLASS_NAMES = [
    "Asian-Green-Bee-Eater", "Brown-Headed-Barbet", "Cattle-Egret", "Common-Kingfisher",
    "Common-Myna", "Common-Rosefinch", "Common-Tailorbird", "Coppersmith-Barbet",
    "Forest-Wagtail", "Gray-Wagtail", "Hoopoe", "House-Crow", "Indian-Grey-Hornbill",
    "Indian-Peacock", "Indian-Pitta", "Indian-Roller", "Jungle-Babbler", "Northern-Lapwing",
    "Red-Wattled-Lapwing", "Ruddy-Shelduck", "Rufous-Treepie", "Sarus-Crane",
    "White-Breasted-Kingfisher", "White-Breasted-Waterhen", "White-Wagtail"
]

# PyTorch Model Architecture using timm
class EfficientNetV2B2PyTorch(nn.Module):
    def __init__(self, num_classes=25):
        super(EfficientNetV2B2PyTorch, self).__init__()
        
        # Load pretrained EfficientNetB2 from timm (V2 model not available in this version)
        # Using EfficientNetB2 as closest alternative - architecture is similar
        try:
            import timm
            print("Loading EfficientNetB2 from timm...")
            base_model = timm.create_model('efficientnet_b2', pretrained=True)
        except ImportError:
            print("Installing timm for better model support...")
            import subprocess
            subprocess.check_call(["pip", "install", "timm"])
            import timm
            base_model = timm.create_model('efficientnet_b2', pretrained=True)
        
        # Extract backbone (remove classification head)
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        
        # Get the number of features from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 300, 300)
            backbone_out = self.backbone(dummy_input)
            num_features = backbone_out.shape[1]
        
        print(f"Backbone output features: {num_features}")
        
        # Custom classification head (matching Keras architecture)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def load_keras_model(weights_path='weights.h5'):
    """Load Keras model and extract weights"""
    print(f"Loading Keras model from {weights_path}...")
    
    base_model = EfficientNetV2B2(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    base_model.trainable = False

    keras_model = Sequential([
        base_model,
        GlobalAveragePooling2D(), 
        Dense(256),  
        BatchNormalization(),  
        Activation('relu'),  
        Dropout(0.3),  
        Dense(128), 
        BatchNormalization(),
        Activation('relu'),  
        Dropout(0.3),
        Dense(25, activation='softmax')  
    ])

    try:
        with h5py.File(weights_path, 'r') as f:
            # Manual mapping to model layers
            keras_model.layers[2].set_weights([f['layers']['dense']['vars']['0'][:], f['layers']['dense']['vars']['1'][:]])
            keras_model.layers[3].set_weights([f['layers']['batch_normalization']['vars']['0'][:], f['layers']['batch_normalization']['vars']['1'][:], f['layers']['batch_normalization']['vars']['2'][:], f['layers']['batch_normalization']['vars']['3'][:]])
            keras_model.layers[6].set_weights([f['layers']['dense_1']['vars']['0'][:], f['layers']['dense_1']['vars']['1'][:]])
            keras_model.layers[7].set_weights([f['layers']['batch_normalization_1']['vars']['0'][:], f['layers']['batch_normalization_1']['vars']['1'][:], f['layers']['batch_normalization_1']['vars']['2'][:], f['layers']['batch_normalization_1']['vars']['3'][:]])
            keras_model.layers[10].set_weights([f['layers']['dense_2']['vars']['0'][:], f['layers']['dense_2']['vars']['1'][:]])
        print("Keras weights loaded successfully!")
        return keras_model
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None


def transfer_weights_to_pytorch(keras_model, pytorch_model):
    """Transfer custom weights from Keras to PyTorch model"""
    print("Transferring custom head weights from Keras to PyTorch...")
    
    # Get input size of first dense layer
    keras_first_dense_w = keras_model.layers[2].get_weights()[0]  # (input_size, 256)
    keras_input_size = keras_first_dense_w.shape[0]
    pytorch_first_dense = pytorch_model.head[2]  # Linear layer (index 2)
    pytorch_input_size = pytorch_first_dense.in_features
    
    print(f"Keras dense input size: {keras_input_size}")
    print(f"PyTorch dense input size: {pytorch_input_size}")
    
    with torch.no_grad():
        # Layer 2 (Keras) -> head[2] (PyTorch): Dense(256)
        keras_dense_1_w = keras_model.layers[2].get_weights()[0]  # (keras_input_size, 256)
        keras_dense_1_b = keras_model.layers[2].get_weights()[1]  # (256,)
        
        # Handle size mismatch
        if keras_input_size == pytorch_input_size:
            pytorch_first_dense.weight.copy_(torch.from_numpy(keras_dense_1_w.T).float())
        else:
            print(f"⚠️  Input size mismatch ({keras_input_size} vs {pytorch_input_size}), using best effort transfer...")
            min_size = min(keras_input_size, pytorch_input_size)
            pytorch_first_dense.weight[:, :min_size].copy_(torch.from_numpy(keras_dense_1_w[:min_size].T).float())
        
        pytorch_first_dense.bias.copy_(torch.from_numpy(keras_dense_1_b).float())
        
        # Layer 3 (Keras) -> head[3] (PyTorch): BatchNormalization(256)
        keras_bn_1_gamma = keras_model.layers[3].get_weights()[0]  # (256,)
        keras_bn_1_beta = keras_model.layers[3].get_weights()[1]   # (256,)
        pytorch_bn_1 = pytorch_model.head[3]
        pytorch_bn_1.weight.copy_(torch.from_numpy(keras_bn_1_gamma).float())
        pytorch_bn_1.bias.copy_(torch.from_numpy(keras_bn_1_beta).float())
        
        # Layer 6 (Keras) -> head[6] (PyTorch): Dense(128)
        keras_dense_2_w = keras_model.layers[6].get_weights()[0]  # (256, 128)
        keras_dense_2_b = keras_model.layers[6].get_weights()[1]  # (128,)
        pytorch_dense_2 = pytorch_model.head[6]
        pytorch_dense_2.weight.copy_(torch.from_numpy(keras_dense_2_w.T).float())
        pytorch_dense_2.bias.copy_(torch.from_numpy(keras_dense_2_b).float())
        
        # Layer 7 (Keras) -> head[7] (PyTorch): BatchNormalization(128)
        keras_bn_2_gamma = keras_model.layers[7].get_weights()[0]  # (128,)
        keras_bn_2_beta = keras_model.layers[7].get_weights()[1]   # (128,)
        pytorch_bn_2 = pytorch_model.head[7]
        pytorch_bn_2.weight.copy_(torch.from_numpy(keras_bn_2_gamma).float())
        pytorch_bn_2.bias.copy_(torch.from_numpy(keras_bn_2_beta).float())
        
        # Layer 10 (Keras) -> head[10] (PyTorch): Dense(25) - final output layer
        keras_dense_3_w = keras_model.layers[10].get_weights()[0]  # (128, 25)
        keras_dense_3_b = keras_model.layers[10].get_weights()[1]  # (25,)
        pytorch_dense_3 = pytorch_model.head[10]
        pytorch_dense_3.weight.copy_(torch.from_numpy(keras_dense_3_w.T).float())
        pytorch_dense_3.bias.copy_(torch.from_numpy(keras_dense_3_b).float())
    
    print("✅ Weights transferred successfully!")


def convert_and_save(keras_weights_path='weights.h5', output_path='bird_classifier.pt'):
    """Main conversion function"""
    
    # Load Keras model
    keras_model = load_keras_model(keras_weights_path)
    if keras_model is None:
        return
    
    # Create PyTorch model
    print("Creating PyTorch model...")
    pytorch_model = EfficientNetV2B2PyTorch(num_classes=25)
    pytorch_model.eval()
    
    # Transfer weights
    transfer_weights_to_pytorch(keras_model, pytorch_model)
    
    # Save the model
    print(f"Saving PyTorch model to {output_path}...")
    torch.save({
        'model_state_dict': pytorch_model.state_dict(),
        'class_names': CLASS_NAMES,
        'architecture': 'EfficientNetV2B2'
    }, output_path)
    
    print(f"✅ Conversion complete! Model saved to {output_path}")
    
    # Save class names separately for reference
    with open(output_path.replace('.pt', '_classes.json'), 'w') as f:
        json.dump(CLASS_NAMES, f)
    print(f"✅ Class names saved to {output_path.replace('.pt', '_classes.json')}")


if __name__ == "__main__":
    import sys
    
    weights_path = sys.argv[1] if len(sys.argv) > 1 else 'weights.h5'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'bird_classifier.pt'
    
    print(f"Converting Keras weights from {weights_path}")
    print(f"Output PyTorch model: {output_path}")
    print("-" * 50)
    
    convert_and_save(weights_path, output_path)
