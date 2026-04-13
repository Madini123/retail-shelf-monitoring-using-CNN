import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import dataset

OUTPUTS_DIR = Path(r"C:\Users\DELL\Desktop\Data Science and Python\outputs")

def denormalize(tensor):
    """
    Reverses the PyTorch/ImageNet normalization done in dataset.py 
    so the image looks normal to the human eye again for verification.
    """
    
    img = tensor.permute(1, 2, 0).numpy()
    

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    
    return np.ascontiguousarray((img * 255).astype(np.uint8))

def run():
    print("====================================")
    print(" Albumentations Verification Render")
    print("====================================\n")
    
    csv_path = r"C:\Users\DELL\Desktop\Data Science and Python\data\SKU110K_fixed\annotations\annotations_train.csv"
    img_dir = r"C:\Users\DELL\Desktop\Data Science and Python\data\SKU110K_fixed\images"
    
    print("Loading PyTorch Custom Dataset...")
    try:
        loader = dataset.test_loader(csv_path, img_dir)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
        
    print("Extracting a randomized augmented batch...")
    
    iterator = iter(loader)
    try:
        images, bboxes = next(iterator)
    except StopIteration:
        print("DataLoader is empty.")
        return

    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(images.size(0)):
        img_np = denormalize(images[i])
        
        axes[i].imshow(img_np)
        
        h, w = 640, 640
        
        boxes_for_image = bboxes[i] 
        
        for box in boxes_for_image:
            
            x_c, y_c, bw, bh, _ = box
            
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2) 
        
        axes[i].imshow(img_np)
        axes[i].axis('off')
        axes[i].set_title(f"Augmented Image {i+1} ({len(boxes_for_image)} items)")
            
    plt.tight_layout()
    
    OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
    out_path = OUTPUTS_DIR / "augmented_batch_test.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n✅ Augmented batch verification saved successfully to: {out_path}")
