import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
xray_base_dir = "./model/xray_labeled"  # Folder containing 'progressor' and 'non_progressor'
clinical_file = "./model/clinical_cleaned.xlsx"

# Load clinical data
clinical_df = pd.read_excel(clinical_file)

# Convert 'ID' to string for matching with filenames
clinical_df["ID"] = clinical_df["ID"].astype(str)

# Normalize label format
clinical_df["Label"] = clinical_df["Label"].astype(float)

# Define image transformations (ResNet50 expects 224x224 images)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset Class
class KneeDataset(Dataset):
    def __init__(self, clinical_df, xray_base_dir, transform=None):
        self.clinical_df = clinical_df
        self.xray_base_dir = xray_base_dir
        self.transform = transform

        # Get list of all image files
        self.image_files = []
        for label in ["progressor", "non_progressor"]:
            label_dir = os.path.join(xray_base_dir, label)
            for img_file in os.listdir(label_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.image_files.append((img_file, label))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file, label = self.image_files[idx]
        patient_id = img_file.split("_")[0]  # Extract patient ID from filename

        # Load X-ray image
        img_path = os.path.join(self.xray_base_dir, label, img_file)
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get corresponding clinical data
        patient_data = self.clinical_df[self.clinical_df["ID"] == patient_id]
        if patient_data.empty:
            raise ValueError(f"Clinical data for patient {patient_id} not found!")

        clinical_features = patient_data.iloc[:, 2:].values.astype(np.float32)  # Exclude 'ID' and 'Label'
        label = np.array([patient_data.iloc[0]["Label"]], dtype=np.float32)  # Convert label to float tensor

        return image, torch.tensor(clinical_features).squeeze(), torch.tensor(label)

# Create dataset and dataloader
dataset = KneeDataset(clinical_df, xray_base_dir, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define MultiModal Model
class MultiModalModel(nn.Module):
    def __init__(self, num_clinical_features, num_classes=1):
        super(MultiModalModel, self).__init__()

        # CNN for X-ray images (ResNet50)
        base_model = models.resnet50(pretrained=True)
        base_model.fc = nn.Identity()  # Remove last layer
        self.cnn = base_model
        self.cnn_fc = nn.Linear(2048, 128)  # Reduce to 128D

        # MLP for Clinical Data
        self.mlp = nn.Sequential(
            nn.Linear(num_clinical_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Fusion Layer + Final Classifier
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, xray_img, clinical_data):
        # Extract features from CNN
        xray_features = self.cnn(xray_img)
        xray_features = self.cnn_fc(xray_features)

        # Extract features from MLP
        clinical_features = self.mlp(clinical_data)

        # Concatenate both feature sets
        combined_features = torch.cat((xray_features, clinical_features), dim=1)

        # Final Classification
        output = self.fusion_fc(combined_features)
        return output

# Initialize Model
num_clinical_features = len(clinical_df.columns) - 2  # Exclude 'ID' and 'Label'
model = MultiModalModel(num_clinical_features=num_clinical_features).to(device)

# Define Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # Binary Classification Loss

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for xray_batch, clinical_batch, labels in dataloader:
        xray_batch, clinical_batch, labels = xray_batch.to(device), clinical_batch.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(xray_batch, clinical_batch).squeeze()  # Ensure correct shape
        loss = criterion(outputs, labels.view(-1))  # Reshape labels
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

print("✅ Training Completed!")
model_save_path = "knee_arthritis_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_epochs': num_epochs
}, model_save_path)

print(f"✅ Model saved as '{model_save_path}'")
