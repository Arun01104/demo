# classify.py

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel
import argparse
import os
from collections import OrderedDict

# --- 1. Define the Model Architecture ---
# This class defines the exact PyTorch model structure. It combines the 
# pre-trained CLIP vision backbone with the custom classification head
# that was trained in the DF40 repository.
class DeepfakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the official pre-trained CLIP Vision model. The DF40 paper used
        # "openai/clip-vit-large-patch14".
        self.backbone = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model
        
        # Define the final classification layer. The input dimension (1024) matches
        # the output feature size of this specific CLIP model. The output is 2 (for real/fake).
        self.head = nn.Linear(1024, 2)

    def forward(self, image):
        # Pass the image through the CLIP backbone to get a feature vector.
        features = self.backbone(image)['pooler_output']
        # Pass that feature vector through our custom classification head.
        prediction = self.head(features)
        return prediction


# --- 2. Create the Main Classification Function ---
def classify_image(config_path, weights_path, image_path):
    """
    Loads the DF40 CLIP model and classifies a single image from a file path.
    """
    # Step A: Setup device (use GPU if available, otherwise CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")

    # Step B: Load model configuration from the YAML file
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Step C: Build the model architecture and move it to the device
    print("Building model architecture...")
    model = DeepfakeCLIP().to(device)

    # Step D: Load the fine-tuned weights
    print(f"Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    
    # This logic handles different ways checkpoints can be saved. Sometimes they have a
    # 'state_dict' key, and sometimes they have a 'module.' prefix from multi-GPU training.
    # This makes our script more robust.
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    
    new_weights = OrderedDict()
    for key, value in checkpoint.items():
        new_key = key.replace('module.', '')
        new_weights[new_key] = value
        
    model.load_state_dict(new_weights, strict=True)
    
    # Set the model to evaluation mode (this disables things like dropout).
    model.eval()

    # Step E: Define the image transformation pipeline.
    # These steps (resize, to_tensor, normalize) MUST match the training setup
    # to get accurate predictions. We read the parameters from the config file.
    img_size = config['resolution']
    mean = config['mean']
    std = config['std']
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Step F: Load and preprocess the target image
    print(f"Loading and preprocessing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device) # Add the batch dimension

    # Step G: Run the prediction
    print("Running inference...")
    with torch.no_grad(): # Disable gradient calculation for speed and memory efficiency
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        
        # The model outputs two scores: [prob_for_real, prob_for_fake]
        fake_score = probabilities[1].item()
        label = 'fake' if fake_score > 0.5 else 'real'
            
    return {'label': label, 'fake_score': fake_score}


# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # Setup a simple command-line argument parser to get the image path.
    parser = argparse.ArgumentParser(description="Classify an image as real or fake.")
    parser.add_argument("image_path", type=str, help="The path to the input image file.")
    args = parser.parse_args()

    # Define the paths to your model files, relative to the script location.
    CONFIG_FILE = os.path.join('models', 'clip.yaml')
    WEIGHTS_FILE = os.path.join('models', 'clip_large.pth') # Make sure your weights file is named clip.pth
    IMAGE_FILE = args.image_path

    # Check that all necessary files exist before trying to run.
    if not all([os.path.exists(p) for p in [CONFIG_FILE, WEIGHTS_FILE, IMAGE_FILE]]):
        print("\nERROR: One or more required files not found.")
        print(f"  - Check for config: {CONFIG_FILE}")
        print(f"  - Check for weights: {WEIGHTS_FILE}")
        print(f"  - Check for image: {IMAGE_FILE}")
    else:
        # If all files are found, run the classification.
        result = classify_image(CONFIG_FILE, WEIGHTS_FILE, IMAGE_FILE)

        # Print the final result in a clean, readable format.
        print("\n" + "="*30)
        print("      CLASSIFICATION RESULT")
        print("="*30)
        print(f"  Label: {result['label'].upper()}")
        print(f"  Confidence (Fake Score): {result['fake_score']:.4f}")
        print("="*30)