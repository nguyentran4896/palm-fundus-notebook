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

# Custom dataset for prediction (inference) phase
class PredictionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Get all image files in the directory (sorted for consistency)
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        print(f"Found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load and resize image to 256x256 (as expected by the model)
        image = Image.open(img_path).convert("RGB")
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        
        # Convert to numpy array
        image = np.array(image)

        # Apply any test-time transformations (e.g., normalization, to tensor)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, img_name

# Function to load the trained model for inference
def load_model(model_path, device):
    # Instantiate the UNet model with 2 output classes (Atrophy, Disc)
    model = UNet(n_channels=3, n_classes=2).to(device)
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model

# Function to visualize and save the prediction results
def save_prediction(image, preds, save_dir, img_name):
    os.makedirs(save_dir, exist_ok=True)
    # Convert image tensor to numpy and move channel to last dimension
    image = image.cpu().numpy().transpose(1, 2, 0)
    preds = preds.cpu().numpy()  # shape: [2, H, W]

    # Denormalize image (reverse the normalization applied during preprocessing)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    # Define masks for each region
    atrophy_mask = preds[0] < 0.5  # Atrophy
    disc_mask = preds[1] < 0.5     # Disc
    # Background is where neither mask is present
    background_mask = ~(atrophy_mask | disc_mask)

    # Highlight only the atrophy (blue) and disc (red) regions on the original image
    overlay = image.copy()
    alpha = 0.4  # Transparency for overlay
    # Atrophy region (blue)
    overlay[atrophy_mask] = (1 - alpha) * image[atrophy_mask] + alpha * np.array([0, 0, 1])
    # Disc region (red)
    overlay[disc_mask] = (1 - alpha) * image[disc_mask] + alpha * np.array([1, 0, 0])
    # Background remains as the original image (no overlay)

    # Create a mask for the retina region (where the original image is not near black)
    retina_mask = (image > 0.05).any(axis=-1)  # shape: (H, W), True inside retina

    # Plot original, overlay, and each mask separately
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    titles = ['Original', 'Overlay', 'Atrophy', 'Disc']
    axes[0].imshow(image)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    for i in range(2):
        axes[i+2].imshow(preds[i], cmap='gray')
        axes[i+2].set_title(titles[i+2])
        axes[i+2].axis('off')

    # Add legend to overlay plot
    patches = [
        mpatches.Patch(color=[0, 0, 1], label='Atrophy'),
        mpatches.Patch(color=[1, 0, 0], label='Disc'),
    ]
    axes[1].legend(handles=patches, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{img_name}'))
    plt.close()

    # Print unique values in mask_sum for debugging
    print("mask_sum unique values:", np.unique(overlay.sum(axis=-1)))

# Main function to run the prediction pipeline
def main():
    # Set device (MPS for Mac, CUDA for GPU, else CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # Define paths for test images and where to save results
    test_image_dir = "self-test-one"
    save_dir = "self-test-one-result"
    
    # test_image_dir = "PALM/Testing/Images"
    # save_dir = "PALM/Testing/Results"
    
    model_path = "best_model.pth"

    # Define test-time transformations (normalization and tensor conversion)
    test_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Create test dataset and dataloader
    test_dataset = PredictionDataset(test_image_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load the trained model
    model = load_model(model_path, device)
    print("Model loaded successfully!")

    # Prediction loop (no gradients needed)
    model.eval()
    
    with torch.no_grad():
        for images, img_names in tqdm(test_loader):
            images = images.to(device)
            
            # Get model outputs (logits after sigmoid, shape: [1, 2, H, W])
            outputs = model(images)
            print(outputs)  # Print raw outputs for debugging
            # Apply threshold to get binary masks for each class
            print(f"Output shape: {outputs.shape}")
            print(f"Output min/max values: {outputs.min():.3f}/{outputs.max():.3f}")
            print(f"Output mean values per class:")
            print(f"  Class 0: {outputs[0,0].mean():.3f}")
            print(f"  Class 1: {outputs[0,1].mean():.3f}")
            predictions = (outputs[0] > 0.5).float()  # shape: [2, H, W]
            
            # Save and visualize predictions for both masks
            save_prediction(images[0], predictions, save_dir, img_names[0])

    print(f"\nPredictions saved in {save_dir} directory")

if __name__ == "__main__":
    main()
