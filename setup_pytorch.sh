#!/bin/bash
# Quick PyTorch Conversion and Inference Setup

echo "======================================"
echo "PyTorch Bird Classifier Converter"
echo "======================================"
echo ""

# Check if weights.h5 exists
if [ ! -f "weights.h5" ]; then
    echo "❌ Error: weights.h5 not found in current directory!"
    echo "Please make sure weights.h5 is in the same folder as this script."
    exit 1
fi

echo "✅ Found weights.h5"
echo ""

# Step 1: Check Python and dependencies
echo "Checking Python dependencies..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')" 2>/dev/null || {
    echo "❌ PyTorch not found. Installing..."
    pip install torch torchvision
}

echo ""

# Step 2: Convert model
echo "Converting Keras model to PyTorch..."
echo "This may take a few minutes on first run..."
echo ""

python convert_to_pytorch.py weights.h5 bird_classifier.pt

if [ $? -ne 0 ]; then
    echo "❌ Conversion failed!"
    exit 1
fi

echo ""
echo "✅ Conversion complete!"
echo ""

# Step 3: Offer to run inference
read -p "Do you want to start inference now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter webcam index (default 0): " webcam_idx
    webcam_idx=${webcam_idx:-0}
    
    read -p "Enter class to suppress (optional, press Enter to skip): " suppress_class
    
    if [ -z "$suppress_class" ]; then
        python bird_classifier_pytorch.py bird_classifier.pt $webcam_idx
    else
        python bird_classifier_pytorch.py bird_classifier.pt $webcam_idx "$suppress_class"
    fi
else
    echo ""
    echo "To run inference later, use:"
    echo "  python bird_classifier_pytorch.py bird_classifier.pt [webcam_index] [suppress_class]"
    echo ""
    echo "Examples:"
    echo "  python bird_classifier_pytorch.py bird_classifier.pt"
    echo "  python bird_classifier_pytorch.py bird_classifier.pt 0"
    echo "  python bird_classifier_pytorch.py bird_classifier.pt 0 'Northern-Lapwing'"
fi
