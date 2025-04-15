# =====================================================================================
#  Çok Modlu Füzyon Modeli: Intermediate Fusion + Multi-Task Learning
# =====================================================================================
# Bu kod, metin ve resim verilerini birleştirmek için Intermediate Fusion (Orta Seviye
# Birleştirme) ve Multi-Task Learning (Çoklu Görev Öğrenimi) yöntemlerini kullanır.
#
# Mimari:
#   1. Model A (Metin): RoBERTa + BiGRU
#   2. Model B (Resim):  ViT + ResNet50 + CBAM
#   3. Intermediate Fusion: Model A ve Model B'den elde edilen özellik vektörleri,
#      modellerin çıkış katmanlarından SONRA, ancak sınıflandırma katmanlarından ÖNCE
#      birleştirilir (concatenation).
#   4. Multi-Task Learning: Birleştirilmiş özellik vektörü, İKİ AYRI sınıflandırıcıya
#      girdi olarak verilir:
#       - fc_text: Metin etiketlerini tahmin eder.
#       - fc_image: Resim etiketlerini tahmin eder.
#      Model, AYNI ANDA hem metin hem de resim etiketlerini tahmin etmeye çalışır.
#   5. Kayıp Fonksiyonu: Metin ve resim için ayrı ayrı kayıplar hesaplanır ve
#      ortalamaları alınır.
#
# Temel Özellikler:
#   - Intermediate Fusion: Özellikler, modellerin iç katmanlarında birleştirilir.
#   - Multi-Task Learning: Model, aynı anda iki görevi (metin ve resim sınıflandırması) öğrenir.
#   - Farklı Etiketler: Metin ve resim için farklı etiket setleri kullanılabilir.
#   - Ayrı Sınıflandırıcılar: Her görev için ayrı bir sınıflandırıcı kullanılır, ancak
#     her ikisi de BİRLEŞTİRİLMİŞ özellikleri girdi olarak alır.
#   - RoBERTa, BiGRU, ViT, ResNet50, CBAM gibi güçlü modeller ve dikkat mekanizması içerir.
#   - Veri Artırma (Data Augmentation): Albumentations kütüphanesi ile resimler üzerinde
#     veri artırma işlemleri yapılır.
#   - AMP (Automatic Mixed Precision): Eğitim süresini hızlandırmak için kullanılır (isteğe bağlı).
#
# Kullanım:
#   - Farklı etiketlere sahip metin ve resim verilerini sınıflandırmak için.
#   - İki modalite (metin ve resim) arasında bir ilişki olduğu varsayıldığında,
#     bu ilişkiyi modellemek ve performansı artırmak için.
#
# =====================================================================================
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO
import base64
from tqdm import tqdm
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from timm import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
from sklearn.metrics import classification_report
import numpy as np

warnings.filterwarnings("ignore")

# === CUDA Ayarları ===
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(42)
    print("CUDA Belleği Temizlendi ve Ayarlandı.")

# === Konfigürasyon ===
A_MAX_LEN = 150
A_BATCH_SIZE = 8  # Model A (metin) için batch size
A_EPOCHS = 15
A_LEARNING_RATE = 2e-5  # Model A'nın learning rate'i.  İki model için farklı lr kullanabilirsiniz.
A_PATIENCE = 2
A_CLASSES = 3  # Metin çıkış sınıf sayısı
A_DROPOUT = 0.4

B_BATCH_SIZE = 16 # Model B (görüntü) için batch size
B_EPOCHS = 15        # Model B için epoch sayısı
B_LEARNING_RATE = 2e-5 # 0.00002 # Model B'nin learning rate'i
B_PATIENCE = 3
B_IMG_SIZE = 224
B_MEAN = [0.485, 0.456, 0.406]
B_STD = [0.229, 0.224, 0.225]
B_CLASSES = 3  # Görüntü çıkış sınıf sayısı


# === Tokenizer (Roberta) ===
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# === Albumentations Dönüşümleri ===
train_transform = A.Compose([
    A.Resize(B_IMG_SIZE, B_IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.Normalize(mean=B_MEAN, std=B_STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(B_IMG_SIZE, B_IMG_SIZE),
    A.Normalize(mean=B_MEAN, std=B_STD),
    ToTensorV2(),
])

# === MultiModalDataset ===
class MultiModalDataset(Dataset):
    def __init__(self, texts, images, text_labels, image_labels, image_transform=None):
        self.texts = texts
        self.images = images
        self.text_labels = text_labels
        self.image_labels = image_labels
        self.image_transform = image_transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if not isinstance(text, str):
            raise ValueError(f"Text at index {idx} is not a string. It is of type {type(text)}.")

        image = Image.open(BytesIO(base64.b64decode(self.images[idx]))).convert("RGB")
        text_label = self.text_labels[idx]
        image_label = self.image_labels[idx]

        text_encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=A_MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        if self.image_transform:
            image = self.image_transform(image=np.array(image))['image']

        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'image': image,
            'text_label': torch.tensor(text_label, dtype=torch.long),
            'image_label': torch.tensor(image_label, dtype=torch.long),
        }

# === Model A: Roberta + BiGRU => 512 boyutlu özellik ===
class RobertaBiGRU(nn.Module):  # Model A'nın adını kısaltıyorum
    def __init__(self, roberta_dim=768, gru_hidden_dim=256, output_dim=512, dropout=0.4):
        super(RobertaBiGRU, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.bigru = nn.GRU(
            input_size=roberta_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(gru_hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = roberta_outputs.last_hidden_state
        bigru_output, _ = self.bigru(sequence_output)
        bigru_pooled = torch.mean(bigru_output, dim=1)  # Ortalama alma
        features = self.fc(bigru_pooled)
        features = self.relu(features)
        return self.dropout(features)

# === CBAM Bloğu ===
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attention = self.channel_attention(x) * x
        avg_pool = torch.mean(channel_attention, dim=1, keepdim=True)
        max_pool, _ = torch.max(channel_attention, dim=1, keepdim=True)
        spatial_attention = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1)) * channel_attention
        return spatial_attention

# === Model B: ViT + ResNet + CBAM => 512 boyutlu özellik ===
class ViTResNetCBAM(nn.Module): # Model B'nin adını kısaltıyorum
    def __init__(self):
        super(ViTResNetCBAM, self).__init__()
        self.vit = create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.vit_fc = nn.Linear(768, 256)
        self.resnet = create_model("resnet50", pretrained=True, num_classes=0)
        self.resnet_fc = nn.Linear(2048, 256)
        self.cbam = CBAM(channels=256 + 256)
        self.fc_out = nn.Linear(256 + 256, 512)

    def forward(self, x):
        vit_features = self.vit(x)
        vit_features = self.vit_fc(vit_features)
        resnet_features = self.resnet(x)
        resnet_features = self.resnet_fc(resnet_features)
        combined_features = torch.cat((vit_features, resnet_features), dim=1)
        combined_features = combined_features.unsqueeze(-1).unsqueeze(-1)  # CBAM için şekli ayarla
        cbam_features = self.cbam(combined_features)
        cbam_features = cbam_features.squeeze(-1).squeeze(-1)  # Fazladan boyutları kaldır
        return self.fc_out(cbam_features)

# === Intermediate Fusion Model (Çoklu Görev Öğrenimi) ===
class IntermediateFusionModel(nn.Module):
    def __init__(self, model_a, model_b, text_classes, image_classes):
        super(IntermediateFusionModel, self).__init__()
        self.model_a = model_a  # RoBERTa + BiGRU
        self.model_b = model_b  # ViT + ResNet + CBAM
        # Intermediate fusion layer (örnek olarak concatenation)
        self.fusion_layer = nn.Linear(512 + 512, 512) # Boyutu ayarlayın. Concatenation sonrası 512+512=1024 olur.
        self.fc_text = nn.Linear(512, text_classes)   # Metin için sınıflandırıcı
        self.fc_image = nn.Linear(512, image_classes) # Resim için sınıflandırıcı
        self.relu = nn.ReLU()


    def forward(self, input_ids, attention_mask, image):
        a_output = self.model_a(input_ids, attention_mask)  # [B, 512]
        b_output = self.model_b(image)  # [B, 512]
        combined = torch.cat((a_output, b_output), dim=1)  # [B, 1024]
        fused = self.relu(self.fusion_layer(combined)) # [B, 512]  # Fusion katmanı ve ReLU
        text_logits = self.fc_text(fused)
        image_logits = self.fc_image(fused)
        return text_logits, image_logits # İki ayrı logits döndür

# === Eğitim Fonksiyonu (Intermediate Fusion, Çoklu Görev) ===
def train_intermediate_fusion(model, train_loader, val_loader, optimizer, criterion, epochs=15, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_train_loss = 0.0
        correct_train_text = 0
        correct_train_image = 0
        total_train_samples = 0

        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            text_labels = batch['text_label'].to(device)
            image_labels = batch['image_label'].to(device)

            text_logits, image_logits = model(input_ids, attention_mask, image) # İki ayrı logits

            loss_text = criterion(text_logits, text_labels)
            loss_image = criterion(image_logits, image_labels)
            loss = (loss_text + loss_image) / 2  # Ortalama kayıp

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            _, text_preds = torch.max(text_logits, dim=1)
            _, image_preds = torch.max(image_logits, dim=1)
            correct_train_text += (text_preds == text_labels).sum().item()
            correct_train_image += (image_preds == image_labels).sum().item()
            total_train_samples += len(text_labels)  # Veya len(image_labels), ikisi de aynı.

        avg_train_loss = total_train_loss / len(train_loader)
        train_text_acc = correct_train_text / total_train_samples
        train_image_acc = correct_train_image / total_train_samples
        print(f"Train Loss: {avg_train_loss:.4f}, Text Acc: {train_text_acc:.4f}, Image Acc: {train_image_acc:.4f}")

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        correct_val_text = 0
        correct_val_image = 0
        total_val_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                image = batch['image'].to(device)
                text_labels = batch['text_label'].to(device)
                image_labels = batch['image_label'].to(device)

                text_logits, image_logits = model(input_ids, attention_mask, image)  # İki ayrı logits

                loss_text = criterion(text_logits, text_labels)
                loss_image = criterion(image_logits, image_labels)
                loss = (loss_text + loss_image) / 2  # Ortalama Kayıp

                total_val_loss += loss.item()

                _, text_preds = torch.max(text_logits, dim=1)
                _, image_preds = torch.max(image_logits, dim=1)
                correct_val_text += (text_preds == text_labels).sum().item()
                correct_val_image += (image_preds == image_labels).sum().item()
                total_val_samples += len(text_labels)  # veya len(image_labels)

        avg_val_loss = total_val_loss / len(val_loader)
        val_text_acc = correct_val_text / total_val_samples
        val_image_acc = correct_val_image / total_val_samples
        print(f"Val Loss: {avg_val_loss:.4f}, Text Acc: {val_text_acc:.4f}, Image Acc: {val_image_acc:.4f}")

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "intermediate_fusion_multitask_best.pth") # Farklı bir isim verin
            print("Model kaydedildi (Validation Loss iyileşti).")
        else:
            patience_counter += 1
            print(f"Validation Loss iyileşmedi. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping tetiklendi.")
                break
# === Test Fonksiyonu (Intermediate Fusion, Çoklu Görev) ===
def test_intermediate_fusion(model, test_loader, criterion): # Fonksiyon adını değiştirin
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_text_labels = []
    all_text_preds = []
    all_image_labels = []
    all_image_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            text_labels = batch['text_label'].to(device)
            image_labels = batch['image_label'].to(device)

            text_logits, image_logits = model(input_ids, attention_mask, image)

            loss_text = criterion(text_logits, text_labels)
            loss_image = criterion(image_logits, image_labels)
            loss = (loss_text + loss_image) / 2
            total_loss += loss.item()


            _, text_preds = torch.max(text_logits, dim=1)
            _, image_preds = torch.max(image_logits, dim=1)

            all_text_labels.extend(text_labels.cpu().numpy())
            all_text_preds.extend(text_preds.cpu().numpy())
            all_image_labels.extend(image_labels.cpu().numpy())
            all_image_preds.extend(image_preds.cpu().numpy())

    text_report = classification_report(all_text_labels, all_text_preds, digits=4)
    print("=== Test (Text) Classification Report ===")
    print(text_report)

    image_report = classification_report(all_image_labels, all_image_preds, digits=4)
    print("=== Test (Image) Classification Report ===")
    print(image_report)

    return text_report, image_report # İki raporu da döndür


# === Ana Program (Main) ===
if __name__ == "__main__":
    # Veri okuma
    df = pd.read_csv("/kaggle/input/sonset/merged_output.csv")  # Kendi dosya yolunuzla değiştirin
    texts = df['metin'].tolist()
    images = df['base64_encoded_image'].tolist()
    text_labels = df['label'].tolist()  # Metin etiketleri
    image_labels = df['kmeans_label'].tolist()  # Resim etiketleri (farklı)

    # Dataset & DataLoader
    train_dataset = MultiModalDataset(texts, images, text_labels, image_labels, image_transform=train_transform)
    val_dataset = MultiModalDataset(texts, images, text_labels, image_labels, image_transform=val_transform)
    # Batch size'ları ayrı ayrı veya aynı tutabilirsiniz.
    train_loader = DataLoader(train_dataset, batch_size=A_BATCH_SIZE, shuffle=True)  # A_BATCH_SIZE kullanıldı
    val_loader = DataLoader(val_dataset, batch_size=B_BATCH_SIZE)       # B_BATCH_SIZE kullanıldı

    # Modeller
    model_a = RobertaBiGRU(output_dim=512)  # Model A
    model_b = ViTResNetCBAM()  # Model B
    # Intermediate Fusion Model (Çoklu Görev)
    fusion_model = IntermediateFusionModel(model_a, model_b, text_classes=A_CLASSES, image_classes=B_CLASSES)

    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=A_LEARNING_RATE) # A_LEARNING_RATE veya B_LEARNING_RATE veya ikisinin ortalaması...
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss


    # Eğitim (Intermediate Fusion, Çoklu Görev)
    train_intermediate_fusion(
        model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=A_EPOCHS,  # A_EPOCHS veya B_EPOCHS kullanabilirsiniz.
        patience=A_PATIENCE  # A_PATIENCE veya B_PATIENCE
    )

    # Test
    fusion_model.load_state_dict(torch.load("intermediate_fusion_multitask_best.pth"))  # Kaydettiğiniz modelin adı
    test_intermediate_fusion(fusion_model, val_loader, criterion)