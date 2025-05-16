import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
from train import UNet
import matplotlib.patches as mpatches

class PredictionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        print(f"Found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load and resize image to 256x256
        image = Image.open(img_path).convert("RGB")
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        
        # Convert to numpy array
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, img_name

def load_model(model_path, device):
    model = UNet(n_channels=3, n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def save_prediction(image, preds, save_dir, img_name):
    os.makedirs(save_dir, exist_ok=True)
    image = image.cpu().numpy().transpose(1, 2, 0)
    preds = preds.cpu().numpy()  # shape: [3, H, W]

    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    # Create color overlays
    overlay = image.copy()
    colors = [
        [1, 0, 0],  # Red for Atrophy
        [0, 1, 0],  # Green for Detachment
        [0, 0, 1],  # Blue for Disc
    ]
    alpha = 0.4  # Transparency for overlay

    # Create a single color mask for all classes (no color addition)
    mask_stack = preds.argmax(axis=0)  # shape: (H, W), values 0,1,2
    mask_sum = np.sum(preds, axis=0)  # shape: (H, W)
    color_mask = np.zeros_like(image)
    for i, color in enumerate(colors):
        for c in range(3):
            color_mask[..., c][mask_stack == i] = color[c]

    # Create a mask for the retina region (where the original image is not near black)
    retina_mask = (image > 0.05).any(axis=-1)  # shape: (H, W), True inside retina

    # Only blend where there is a mask (any class) AND inside the retina
    overlay_mask = (mask_sum > 0) & retina_mask
    overlay = np.where(overlay_mask[..., None], (1 - alpha) * image + alpha * color_mask, image)

    # Plot original, overlay, and each mask
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    titles = ['Original', 'Overlay', 'Atrophy', 'Detachment', 'Disc']
    axes[0].imshow(image)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    for i in range(3):
        axes[i+2].imshow(preds[i], cmap='gray')
        axes[i+2].set_title(titles[i+2])
        axes[i+2].axis('off')

    # Add legend to overlay
    patches = [
        mpatches.Patch(color=colors[0], label='Atrophy'),
        mpatches.Patch(color=colors[1], label='Detachment'),
        mpatches.Patch(color=colors[2], label='Disc'),
    ]
    axes[1].legend(handles=patches, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{img_name}'))
    plt.close()

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
    test_image_dir = "self-test"
    save_dir = "predictions"
    
    # test_image_dir = "PALM/Testing/Images"
    # save_dir = "PALM/Testing/Results"
    
    model_path = "best_model.pth"

    # Define transforms
    test_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Create test dataset and dataloader
    test_dataset = PredictionDataset(test_image_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load model
    model = load_model(model_path, device)
    print("Model loaded successfully!")

    # Prediction loop
    model.eval()
    
    with torch.no_grad():
        for images, img_names in tqdm(test_loader):
            images = images.to(device)
            
            # Get predictions
            outputs = model(images)  # shape: [1, 3, H, W]
            predictions = (outputs > 0.5).float()
            
            # Save predictions for all 3 masks
            save_prediction(images[0], predictions[0], save_dir, img_names[0])

    print(f"\nPredictions saved in {save_dir} directory")

if __name__ == "__main__":
    main()
