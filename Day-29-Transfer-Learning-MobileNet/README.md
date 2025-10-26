# 🦾 Transfer Learning with MobileNet

This project demonstrates how to fine-tune a **pre-trained MobileNet** model for a **custom image classification** task using Keras.


## 🚀 Features
- Uses MobileNet for fast transfer learning
- Fine-tunes the last layers for your dataset
- Visualizes training accuracy and loss
- Easy to adapt for any 2+ class problem


## 🛠️ Installation
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

📂 Dataset

Organize your dataset as:

dataset/
├── train/
│   ├── class1/
│   └── class2/
└── validation/
    ├── class1/
    └── class2/

    Example:

dataset/train/cats
dataset/train/dogs
dataset/validation/cats
dataset/validation/dogs
▶️ Run
python transfer_learning_mobilenet.py

📊 Example Output

Training Accuracy: 95%

Validation Accuracy: 92%

Model saved as mobilenet_finetuned.h5

🧰 Model Used

MobileNetV2 (pretrained on ImageNet)
Fine-tuned on a custom dataset.

✅ How It Works

Loads MobileNetV2 pretrained on ImageNet.

Adds new dense layers for your dataset.

Trains on your data (freezes base model first, then fine-tunes).

Saves your trained model.

Plots accuracy curves.