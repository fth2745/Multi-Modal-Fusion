# Multi-Modal-Fusion

# 🧠 Multi-Modal Intermediate Fusion Model with Multi-Task Learning

This repository contains a deep learning framework that combines **text and image modalities** for joint classification using **Intermediate Fusion** and **Multi-Task Learning** strategies. The model simultaneously predicts **text-based** and **image-based** labels from fashion product data.

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
