# Multi-Modal-Fusion

# 🧠 Multi-Modal Intermediate Fusion Model with Multi-Task Learning

This repository contains a deep learning framework that integrates textual and visual modalities to perform joint classification tasks on fashion product data. Leveraging Intermediate Fusion and Multi-Task Learning strategies, the model extracts high-level features from both text descriptions and product images to predict semantic labels. By combining outputs from RoBERTa-BiGRU (for text) and ViT-ResNet50-CBAM (for images), the system simultaneously classifies modality-specific targets with enhanced accuracy. The project demonstrates how multimodal fusion can improve understanding of complex, real-world data where visual and textual cues complement each other

---

## 🔍 Overview

- **Modality A (Text):** RoBERTa + BiGRU
- **Modality B (Image):** ViT + ResNet-50 + CBAM (attention module)
- **Fusion Strategy:** Intermediate fusion (after feature extraction, before classification)
- **Learning Strategy:** Multi-task learning with two separate classifiers:
  - `fc_text`: Predicts labels based on textual descriptions
  - `fc_image`: Predicts labels from product images

---

## 🧱 Architecture

```text
         ┌────────────┐      ┌────────────┐
         │ Text Input │      │ Image Input│
         └─────┬──────┘      └────┬───────┘
               ↓                     ↓
     [RoBERTa + BiGRU]         [ViT + ResNet + CBAM]
               ↓                     ↓
         Text Features           Image Features
               └────┬──────Concatenation──────┬────┘
                    ↓                         ↓
              Intermediate Fusion Layer (Linear + ReLU)
                    ↓
        ┌────────────┴────────────┐
        ↓                         ↓
    Text Classifier         Image Classifier
        ↓                         ↓
   Text Prediction           Image Prediction
