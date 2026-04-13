import os
import tarfile
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path
import shutil
import glob


DATA_DIR = Path(r"C:\Users\DELL\Desktop\Data Science and Python\data")
TAR_FILE = DATA_DIR / "SKU110K_fixed.tar.gz"
EXTRACTED_DIR = DATA_DIR / "SKU110K_fixed"
OUTPUTS_DIR = Path(r"C:\Users\DELL\Desktop\Data Science and Python\outputs")

def extract_dataset():
    """Extracts the dataset if not already extracted."""
    print("--- Stage 1: Dataset Extraction ---")
    if EXTRACTED_DIR.exists():
        print(f"Dataset already extracted at {EXTRACTED_DIR}")
        return True
    
    if not TAR_FILE.exists():
        print(f"Error: Tar file not found at {TAR_FILE}")
        return False
        
    print(f"Extracting {TAR_FILE}... This might take a while for ~12GB.")
    
    os.system(f'tar -xf "{TAR_FILE}" -C "{DATA_DIR}"')
    
    if EXTRACTED_DIR.exists():
        print("Extraction completed successfully!")
        return True
    else:
        print("Extraction failed. Please extract manually.")
        return False

def discover_annotations():
    """Finds the locations of image and annotation files."""
    images_dir = EXTRACTED_DIR / "images"
    annotations_csv = None
    
    
    possible_csv_paths = [
        EXTRACTED_DIR / "annotations" / "annotations.csv",
        EXTRACTED_DIR / "annotations.csv",
    ]
    for p in possible_csv_paths:
        if p.exists():
            annotations_csv = p
            break
            

    if annotations_csv is None:
        print("Checking for split CSVs...")
        train_csv = EXTRACTED_DIR / "annotations" / "annotations_train.csv"
        
        if train_csv.exists():
           annotations_csv = train_csv 

    return images_dir, annotations_csv

def explore_data(annotations_csv):
    """Explores the dataset: class distribution, image resolution stats."""
    print("\n--- Stage 2: Data Exploration ---")
    if not annotations_csv or not annotations_csv.exists():
        print(f"Could not find annotations.csv at {annotations_csv}")
        return None
        
    print(f"Loading annotations from {annotations_csv}")
    
    
    try:
        df = pd.read_csv(annotations_csv, header=None)
        if len(df.columns) == 8:
            df.columns = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class_id', 'image_width', 'image_height']
            
            print(f"Total annotations (bounding boxes): {len(df)}")
            print(f"Total unique images: {df['image_name'].nunique()}")
            
            # Bounding boxes per image
            boxes_per_img = df.groupby('image_name').size()
            print("\nBounding Boxes per Image Statistics:")
            print(boxes_per_img.describe())
            
            # Resolution stats
            print("\nImage Resolution (Width) Statistics:")
            print(df['image_width'].describe())
            print("\nImage Resolution (Height) Statistics:")
            print(df['image_height'].describe())
            
            # Class distribution
            print("\nClass Distribution:")
            print(df['class_id'].value_counts())
            
            return df
        else:
            print("CSV format doesn't match standard 8-column format.")
            print(df.head())
            return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def convert_to_yolo_format(df, images_dir):
    """Converts the annotations to YOLO format."""
    print("\n--- Stage 3: YOLO Format Conversion ---")
    
    
    # The annotations format is x1, y1, x2, y2 -> x_center, y_center, width, height 
    
    print("Normalizing bounding boxes...")
    
    df['x_center'] = ((df['x1'] + df['x2']) / 2) / df['image_width']
    df['y_center'] = ((df['y1'] + df['y2']) / 2) / df['image_height']
    df['width'] = (df['x2'] - df['x1']) / df['image_width']
    df['height'] = (df['y2'] - df['y1']) / df['image_height']
    

    df['yolo_class'] = 0
    
    
    unique_images = df['image_name'].unique()
    print(f"Prepared YOLO conversion for {len(unique_images)} images.")
    print("Example calculation completed.")
    return df

def visualize_samples(df, images_dir, num_samples=20):
    """Displays random annotated samples to verify labels."""
    print("\n--- Stage 4: Visualizing Samples ---")
    
    unique_images = df['image_name'].unique()
    sample_images = random.sample(list(unique_images), min(num_samples, len(unique_images)))
    
    print(f"Generating visualizations for {len(sample_images)} samples...")
    OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    axes = axes.flatten()
    
    for i, img_name in enumerate(sample_images):
        img_path = images_dir / img_name
        
        if not img_path.exists():
            
            if (images_dir / f"{img_name}.jpg").exists():
                img_path = images_dir / f"{img_name}.jpg"
            else:
                 print(f"Warning: Image {img_path} not found.")
                 continue

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get bounding boxes for this image
        boxes = df[df['image_name'] == img_name]
        
        for _, row in boxes.iterrows():
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            # Draw bounding box 
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{img_name}\n({len(boxes)} items)", fontsize=8)
        
    plt.tight_layout()
    output_path = OUTPUTS_DIR / "annotation_samples.png"
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved successfully to: {output_path}")

def main():
    print("=============================================")
    print("   CNN Project - Data Preparation")
    print("=============================================\n")
    
    success = extract_dataset()
    if not success:
        return
        
    images_dir, annotations_csv = discover_annotations()
    
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return
        
    df = explore_data(annotations_csv)
    if df is not None:
        yolo_df = convert_to_yolo_format(df, images_dir)
        visualize_samples(df, images_dir, num_samples=20)
        
    print("\nData Preparation Check Completed!")

if __name__ == "__main__":
    main()
