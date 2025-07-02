## 🧬 Introduction

This repository presents a Vision Transformer (ViT-B/16) model fine-tuned on the **BACH (Breast Cancer Histology)** dataset for the task of **breast cancer classification**. It demonstrates the power of transformer-based architectures in the field of digital pathology, offering an end-to-end PyTorch pipeline suitable for both research and practical deployment.

Breast cancer is one of the leading causes of cancer-related deaths worldwide, and early detection is vital for effective treatment. Histopathology images, stained using Hematoxylin and Eosin (H&E), are commonly used for diagnosis. However, manual interpretation is time-consuming and subjective. By leveraging **Vision Transformers (ViTs)**—which are pretrained on large datasets like ImageNet and known for their global attention capabilities—we aim to classify these images into four clinically meaningful categories:

- **Benign**
- **In Situ Carcinoma**
- **Invasive Carcinoma**
- **Normal**

This project uses transfer learning by freezing the pretrained ViT backbone and replacing the classification head with a task-specific linear layer. The model is trained and evaluated on a custom split of the BACH dataset, with data augmentation and standardized preprocessing applied.

The codebase is modular, reproducible, and easy to extend. It includes:
- Training and evaluation scripts
- Custom image prediction utility
- Preprocessing and data loading pipeline
- Visualization tools for loss curves and predictions

This project is an excellent starting point for those interested in applying Vision Transformers to medical imaging problems, and can be extended with self-supervised learning (e.g., DINOv2), domain adaptation, or attention-based explainability.

## 📚 Dataset

This project is based on the **BACH (Breast Cancer Histology)** dataset, originally released as part of the **ICIAR 2018 Grand Challenge** on breast cancer image classification. The full official dataset contains:

- **400 microscopy images**, each of size **2048×1536 pixels**
- **4 balanced classes**:  
  - **Normal**, **Benign**, **In Situ Carcinoma**, **Invasive Carcinoma**
- **Annotations**: Per-image expert pathology labels
- **Download size**: ~10.4 GB (training) + ~3 GB (test)
- **License**: CC BY-NC-ND (non-commercial, no derivatives)

📥 **Official Source**:  
[https://iciar2018-challenge.grand-challenge.org/Dataset/](https://iciar2018-challenge.grand-challenge.org/Dataset/)

---

### ⚠️ Note on Dataset Usage

Due to hardware constraints, the full ~13 GB dataset is **computationally intensive** for our current setup. Therefore, we used a **smaller, preprocessed version** of the dataset from Kaggle:

📦 **Used Dataset:**  
**ICIAR2018 70x30 with DA (Data Augmentation)**  
[https://www.kaggle.com/datasets/abdulrahmanamukhlif/iciar2018-70x30-with-da](https://www.kaggle.com/datasets/abdulrahmanamukhlif/iciar2018-70x30-with-da)

- This version contains a **70% training / 30% test** split.
- Data is organized in folders by class.
- Augmented samples are included for improved model generalization.

👉 **Note:** A **small validation set** was managed separately and internally, based on randomized selection, to monitor training performance. Its creation process is anonymized to maintain reproducibility without depending on an official validation split.

---

### 📁 Folder Structure

Your dataset should be organized as follows:

data/
├── train/
│ ├── Benign/
│ │ ├── image_001.png
│ │ ├── image_002.png
│ │ └── ...
│ ├── InSitu/
│ │ ├── image_101.png
│ │ └── ...
│ ├── Invasive/
│ │ ├── image_201.png
│ │ └── ...
│ └── Normal/
│ ├── image_301.png
│ └── ...
│
├── test/
│ ├── Benign/
│ ├── InSitu/
│ ├── Invasive/
│ └── Normal/


## 🧠 Project Architecture

This repository follows a clean, modular architecture to ensure readability, reusability, and scalability. The project is divided into the following components:

### 🔍 1. **Model Architecture**
- Utilizes **Vision Transformer (ViT-B/16)** from `torchvision.models`, pretrained on **ImageNet**.
- The transformer backbone is **frozen**, and a new **classification head** is added:
  - A linear layer outputs logits for 4 classes: `['Benign', 'InSitu', 'Invasive', 'Normal']`.

### 🔄 2. **Training Workflow**
- Training logic is abstracted into `engine/train_eval.py`, which handles:
  - Epoch loops
  - Forward pass
  - Backpropagation
  - Accuracy/loss calculation
- Uses **CrossEntropyLoss** and **Adam optimizer**.
- Seed initialization is handled for full reproducibility.

### 🧾 3. **Data Pipeline**
- Data is loaded using `torchvision.datasets.ImageFolder`.
- Transforms (resizing, normalization, augmentation) are taken from the pretrained ViT weights.
- Batch loading is done with PyTorch's `DataLoader` and auto-parallelism via `num_workers`.

### 📊 4. **Evaluation & Visualization**
- Loss and accuracy curves are plotted using `helper_functions.plot_loss_curves()`.
- Includes a separate `predict.py` script to test the model on custom images with visual output.

### 🧱 5. **Modular Design**
Each component lives in its own folder:
├── models/ → ViT model definition and head
├── engine/ → Training & evaluation logic
├── utils/ → Seed setting, visualization tools
├── scripts/ → Main training pipeline
├── predictions/ → Inference on single images






