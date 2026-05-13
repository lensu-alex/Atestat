# Atestat
Basic Real-Time Weapon Detection using ViT model


# AI Safety Guard 🛡️

AI Safety Guard is a real-time computer vision application that detects potentially dangerous objects using a standard webcam. Instead of relying on traditional object detection models trained on fixed datasets, this project uses **Zero-Shot Learning**, enabling the system to recognize objects directly from natural language descriptions.

The application is designed for environments where fast and automated visual monitoring is important, such as schools, offices, laboratories, or restricted areas.

---

## ✨ Features

- **Real-Time Detection**  
  Detects dangerous objects instantly through a live webcam feed.

- **Zero-Shot Object Recognition**  
  Recognizes objects without needing custom training datasets.

- **Natural Language Labels**  
  Add or remove detectable objects simply by editing text prompts.

- **GPU Acceleration with NVIDIA CUDA**  
  Optimized for high-speed inference and near real-time performance.

- **Flexible & Expandable**  
  Easily extend detection capabilities without retraining the model.

---

## 🎯 Detectable Objects

The system is currently configured to detect:

- ✂️ Scissors  
- 🍴 Forks  
- 🔪 Knives  
- 🔫 Firearms  

Additional objects can be added by modifying the target text labels.

---

## 🧠 How It Works

AI Safety Guard uses a **Zero-Shot Learning** approach powered by vision-language models. Instead of learning only from predefined training classes, the model compares visual features from the webcam with textual descriptions provided by the user.

Example:

```python
["a knife", "a gun", "a pair of scissors", "a fork"]
