import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define the U-Net architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up_conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv4(x)
        
        x = self.outc(x)
        return self.sigmoid(x)

# Custom Dataset class
class PALMDataset(Dataset):
    def __init__(self, image_dir, mask_dirs, transform=None):
        self.image_dir = image_dir
        self.mask_dirs = mask_dirs
        self.transform = transform
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # Build a dict for mask files by base name for each mask type
        mask_dicts = []
        for mask_dir in mask_dirs:
            mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            mask_dicts.append({os.path.splitext(f)[0]: f for f in mask_files})
        
        # Store all valid image-mask pairs, including partial masks
        self.pairs = []
        for img in self.images:
            base = os.path.splitext(img)[0]
            mask_names = []
            mask_exists = []
            
            # Check each mask type
            for mask_dict in mask_dicts:
                if base in mask_dict:
                    mask_names.append(mask_dict[base])
                    mask_exists.append(True)
                else:
                    mask_names.append(None)
                    mask_exists.append(False)
            
            # Only add if at least one mask exists
            if any(mask_exists):
                self.pairs.append((img, mask_names, mask_exists))
        
        print(f"Found {len(self.pairs)} images with at least one mask")
        # Print statistics about mask availability
        total_masks = len(self.pairs) * len(mask_dirs)
        available_masks = sum(sum(1 for exists in pair[2] if exists) for pair in self.pairs)
        print(f"Mask availability: {available_masks}/{total_masks} ({available_masks/total_masks*100:.1f}%)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_names, mask_exists = self.pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load and resize image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        image = np.array(image)
        
        # Initialize masks array with zeros
        masks = np.zeros((len(self.mask_dirs), 256, 256), dtype=np.float32)
        
        # Load and process available masks
        for i, (mask_name, exists) in enumerate(zip(mask_names, mask_exists)):
            if exists:
                mask_path = os.path.join(self.mask_dirs[i], mask_name)
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize((256, 256), Image.Resampling.NEAREST)
                mask = np.array(mask)
                masks[i] = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, masks=masks)
            image = augmented["image"]
            masks = augmented["masks"]

        return image, masks, mask_exists

# Custom loss function for handling missing masks
class WeightedBCELoss(nn.Module):
    def __init__(self, missing_weight=0.0):
        super().__init__()
        self.missing_weight = missing_weight
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(self, pred, target, mask_exists):
        # Calculate BCE loss for all pixels
        loss = self.bce(pred, target)
        # Create weights tensor using broadcasting
        weights = mask_exists.float() + (~mask_exists).float() * self.missing_weight
        weighted_loss = (loss * weights).mean()
        return weighted_loss

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, masks, mask_exists in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        # Convert mask_exists to tensor and reshape for broadcasting
        mask_exists = torch.tensor(np.array(mask_exists), dtype=torch.bool, device=device)
        if mask_exists.shape[0] == masks.shape[1] and mask_exists.shape[1] == masks.shape[0]:
            mask_exists = mask_exists.t()  # transpose if needed
        mask_exists = mask_exists.unsqueeze(-1).unsqueeze(-1)  # shape: [batch, num_classes, 1, 1]
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate weighted loss
        loss = criterion(outputs, masks, mask_exists)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Validation function
def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks, mask_exists in tqdm(val_loader):
            images = images.to(device)
            masks = masks.to(device)
            # Convert mask_exists to tensor and reshape for broadcasting
            mask_exists = torch.tensor(np.array(mask_exists), dtype=torch.bool, device=device)
            if mask_exists.shape[0] == masks.shape[1] and mask_exists.shape[1] == masks.shape[0]:
                mask_exists = mask_exists.t()  # transpose if needed
            mask_exists = mask_exists.unsqueeze(-1).unsqueeze(-1)  # shape: [batch, num_classes, 1, 1]
            
            outputs = model(images)
            loss = criterion(outputs, masks, mask_exists)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # Define paths
    train_image_dir = "PALM/Training/Images"
    train_mask_dirs = [
        "PALM/Training/Lesion Masks/Atrophy",
        "PALM/Training/Lesion Masks/Detachment",
        "PALM/Training/Disc Masks"
    ]
    val_image_dir = "PALM/Validation/Images"
    val_mask_dirs = [
        "PALM/Validation/Lesion Masks/Atrophy",
        "PALM/Validation/Lesion Masks/Detachment",
        "PALM/Validation/Disc Masks"
    ]

    # Define transforms
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Create datasets
    train_dataset = PALMDataset(train_image_dir, train_mask_dirs, transform=train_transform)
    val_dataset = PALMDataset(val_image_dir, val_mask_dirs, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = UNet(n_channels=3, n_classes=3).to(device)  # 3 classes for Atrophy, Detachment, Disc
    criterion = WeightedBCELoss(missing_weight=0.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = validate_model(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model!")

        # Plot training progress
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_progress.png')
        plt.close()

if __name__ == "__main__":
    main()
