# 🧠 Image Autoencoder (Keras)

An **Autoencoder** is a neural network that learns to compress and reconstruct data.
This project demonstrates an image autoencoder trained on the **MNIST dataset**.

---

## 🚀 Features

- Compresses 28x28 MNIST images to a 32-dimensional latent space
- Reconstructs the images from the latent space
- Visualizes original vs reconstructed images
- Simple encoder–decoder architecture using Keras

---

## 🛠️ Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

▶️ Run

python image_autoencoder.py


## 📊 Output


Shows side-by-side comparison of **original** and **reconstructed** images
Displays **training loss curve**

Saves the trained model as `autoencoder_mnist.h5`

---


---
## 🧰 Architecture


* **Encoder:** Dense layers compress input to latent vector
* **Decoder:** Dense layers reconstruct the original image
