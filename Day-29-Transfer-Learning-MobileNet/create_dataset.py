"""
Create a simple synthetic dataset for transfer learning demo.
Generates colorful geometric shapes in different categories.
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import shutil

def create_directory_structure():
    """Create train/validation folder structure."""
    base_dir = "dataset"
    
    # Remove existing dataset if present
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    # Create structure
    categories = ['circles', 'squares']
    splits = ['train', 'validation']
    
    for split in splits:
        for category in categories:
            path = os.path.join(base_dir, split, category)
            os.makedirs(path, exist_ok=True)
    
    return categories, splits

def create_circle_image(size=224):
    """Generate an image with a circle."""
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    
    # Random circle properties
    color = tuple(np.random.randint(50, 200, 3).tolist())
    margin = 40
    draw.ellipse([margin, margin, size-margin, size-margin], fill=color, outline='black', width=3)
    
    return img

def create_square_image(size=224):
    """Generate an image with a square."""
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    
    # Random square properties
    color = tuple(np.random.randint(50, 200, 3).tolist())
    margin = 40
    draw.rectangle([margin, margin, size-margin, size-margin], fill=color, outline='black', width=3)
    
    return img

def generate_dataset():
    """Generate complete dataset."""
    print("🎨 Creating synthetic dataset...")
    
    categories, splits = create_directory_structure()
    
    # Number of images per category per split
    num_train = 100  # 100 images per class for training
    num_val = 30     # 30 images per class for validation
    
    generators = {
        'circles': create_circle_image,
        'squares': create_square_image
    }
    
    for split in splits:
        num_images = num_train if split == 'train' else num_val
        
        for category in categories:
            print(f"  📁 Generating {num_images} {category} images for {split}...")
            
            for i in range(num_images):
                img = generators[category]()
                path = os.path.join('dataset', split, category, f'{category}_{i:03d}.png')
                img.save(path)
    
    print("\n✅ Dataset created successfully!")
    print(f"   📊 Training: {num_train * len(categories)} images ({num_train} per class)")
    print(f"   📊 Validation: {num_val * len(categories)} images ({num_val} per class)")
    print(f"   📂 Location: dataset/")

if __name__ == "__main__":
    generate_dataset()
