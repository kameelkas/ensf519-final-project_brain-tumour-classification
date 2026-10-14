# Brain Tumor MRI Classification Project

A deep learning project developed for **MedAI Diagnostics** to efficiently classify brain tumor MRI scans into four distinct categories: Glioma, Meningioma, Pituitary, and No Tumour.

This repository contains two complete approaches: a custom-built Convolutional Neural Network (CNN) trained from scratch and a transfer learning pipeline leveraging pre-trained **EfficientNet-B0**.

---

## 👥 Project Team & Contributions

This was a collaborative 4-person group project for ENSF 519, with work distributed among team members:

### My Specific Contributions:
* **Transfer Learning & Model Implementation:** Co-authored and focused heavily on the pre-trained EfficientNet model pipeline.
* **Data Pipeline & Preprocessing:** Wrote the code to load and preprocess data, handle normalization, and organize datasets into batched PyTorch `DataLoader` objects.
* **Architecture Customization:** Loaded the pre-trained EfficientNet model, selective unfroze key convolutional layers (specifically fine-tuning the final `Conv2dNormActivation` block), and replaced the original output with a custom classifier head to make both the head and unfrozen layers trainable for our 4-class classification task.
* **Collaboration & Refinement:** Collaborated with team members to integrate validation sets, implement weighted random sampling for class balancing, and incorporate training accuracy metrics and formatted progress logging during execution.

---

## 📊 Dataset & Preprocessing

* **Dataset Source:** Brain Tumour MRI Dataset (via KaggleHub / Masoud Nickparvar) containing 7,023 total MRI scans.
* **Target Classes (4):** Glioma tumour, Meningioma tumour, No tumour, Pituitary tumour.
* **Data Preprocessing & Augmentation:**
  * Standardized image resolution to $224 \times 224$ pixels.
  * Pixel normalization to $[-1, 1]$ (or ImageNet normalization for EfficientNet).
  * Custom bounding-box/contour cropping using OpenCV (in the EfficientNet pipeline) to isolate the brain region.
  * Data augmentation including random rotations ($\pm 10^\circ$) and horizontal flips ($p = 0.5$).
  * Handled class imbalance using PyTorch's `WeightedRandomSampler`.

---

## 🧠 Model Architectures & Strategies

### Model 1: Custom CNN
* **Architecture:** A deep 4-stage convolutional neural network featuring progressive depth (filter count doubles from 32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256).
* **Regularization:** Batch Normalization applied after every convolution for convergence stability, alongside a 0.5 Dropout rate in fully connected layers to prevent overfitting.
* **Optimization:** Trained using the Adam optimizer and a `ReduceLROnPlateau` learning rate scheduler (reducing learning rate by 10% if validation loss plateaus for 3 epochs).
* **Performance:** Achieved an overall **90% test accuracy** (Macro/Weighted F1-score: 0.90).

### Model 2: Transfer Learning with EfficientNet-B0
* **Architecture:** Pre-trained EfficientNet-B0 (~4 million total parameters) with frozen feature extraction layers, an unfrozen final `Conv2dNormActivation` block, and a custom classifier head featuring a 40% dropout layer.
* **Training Setup:** 80/20 train-validation split, cross-entropy loss, Adam optimizer, and `ReduceLROnPlateau` scheduler.
* **Performance:** Achieved a stellar **94.28% test accuracy** and a **0.9427 F1-score**.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.11
* **Frameworks & Libraries:** PyTorch, Torchvision, TorchInfo
* **Image Processing:** OpenCV, Pillow, Imutils
* **Metrics & Visualization:** Scikit-Learn, Matplotlib, Seaborn, TQDM
