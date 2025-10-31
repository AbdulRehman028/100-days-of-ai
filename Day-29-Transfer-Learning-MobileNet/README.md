# 🦾 Transfer Learning with MobileNet

This project demonstrates how to fine-tune a **pre-trained MobileNet** model for a **custom image classification** task using TensorFlow/Keras.

## 🚀 Features
- Uses MobileNet for fast transfer learning
- Fine-tunes the last layers for your dataset
- Visualizes training accuracy and loss
- Easy to adapt for any 2+ class problem
- Includes synthetic dataset generator for quick testing

## 🛠️ Installation

```powershell
pip install -r requirements.txt
```

## 📂 Dataset Setup

### Option 1: Use Synthetic Dataset (Quick Start)
Generate a simple dataset of geometric shapes:

```powershell
python create_dataset.py
```

This creates:
- 200 training images (100 circles, 100 squares)
- 60 validation images (30 circles, 30 squares)

### Option 2: Use Your Own Dataset
Organize your dataset as:
```
dataset/
├── train/
│   ├── class1/
│   └── class2/
└── validation/
    ├── class1/
    └── class2/
```

**Example:**
- `dataset/train/cats`
- `dataset/train/dogs`
- `dataset/validation/cats`
- `dataset/validation/dogs`

## ▶️ Run

```powershell
python transfer_learning_mobilenet.py
```

## 📊 Output

- **Model**: Saved as `mobilenet_finetuned.h5`
- **Visualization**: Training/validation accuracy plot
- **Console**: Epoch-by-epoch training metrics

## 🧰 Model Architecture

- **Base**: MobileNetV2 (pretrained on ImageNet)
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense(128, relu)
  - Dropout(0.3)
  - Dense(num_classes, softmax)

## ✅ How It Works

1. Loads MobileNetV2 pretrained on ImageNet
2. Freezes base model weights
3. Adds custom classification head
4. Trains for 5 epochs (feature extraction)
5. Unfreezes base model
6. Fine-tunes for 3 epochs with low learning rate
7. Saves trained model and plots accuracy

## 🎯 Use Cases

- Custom image classification
- Fine-tuning for specific domains
- Quick prototyping with transfer learning
- Educational demonstrations