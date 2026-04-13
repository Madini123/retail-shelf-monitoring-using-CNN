import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

DATA_DIR = Path(r"C:\Users\DELL\Desktop\Data Science and Python\data")
EXTRACTED_DIR = DATA_DIR / "SKU110K_fixed"

def create_yolo_dirs():
    """Creates the necessary folder structure for YOLO model training."""
    print("Creating YOLO directory structure...")
    splits = ['train', 'val', 'test']
    for split in splits:
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def process_and_copy(df, image_list, split_name):
    """
    Copies images to their respective split folders and generates 
    normalized YOLO .txt annotation files for each image.
    """
    images_source_dir = EXTRACTED_DIR / "images"
    dest_images_dir = DATA_DIR / "images" / split_name
    dest_labels_dir = DATA_DIR / "labels" / split_name
    
    print(f"\nProcessing {len(image_list)} images for the '{split_name}' split...")
    
    # Filter the dataframe to only rows corresponding to this split's images
    split_df = df[df['image_name'].isin(image_list)].copy()
    
    # Pre-calculate YOLO normalized coordinates for speed
    # YOLO format: <class> <x_center> <y_center> <width> <height>
    split_df['x_center'] = ((split_df['x1'] + split_df['x2']) / 2) / split_df['image_width']
    split_df['y_center'] = ((split_df['y1'] + split_df['y2']) / 2) / split_df['image_height']
    split_df['w'] = (split_df['x2'] - split_df['x1']) / split_df['image_width']
    split_df['h'] = (split_df['y2'] - split_df['y1']) / split_df['image_height']
    split_df['yolo_class'] = 0 # 0 for 'object/product'
    
    # Group by image so we can write one TXT file with multiple lines
    grouped = split_df.groupby('image_name')
    
    for image_name, group in tqdm(grouped, desc=f"Writing {split_name} data"):
        # 1. Copy Image
        src_img_path = images_source_dir / image_name
        dst_img_path = dest_images_dir / image_name
        
        # If the image name in CSV doesn't have the extension, handle it:
        if not src_img_path.exists():
            src_img_path = images_source_dir / f"{image_name}.jpg"
            dst_img_path = dest_images_dir / f"{image_name}.jpg"
            
        if src_img_path.exists():
            shutil.copy(src_img_path, dst_img_path)
        else:
            continue
            
        # 2. Write Labels
        # Get purely the file name without extension
        base_name = Path(dst_img_path).stem 
        label_path = dest_labels_dir / f"{base_name}.txt"
        
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                # Format: 0 0.5 0.5 0.2 0.2
                line = f"{int(row['yolo_class'])} {row['x_center']:.6f} {row['y_center']:.6f} {row['w']:.6f} {row['h']:.6f}\n"
                f.write(line)

def run():
    print("====================================")
    print(" YOLO Format Data Preparation Pipeline")
    print("====================================\n")
    
    csv_path = EXTRACTED_DIR / "annotations" / "annotations.csv"
    if not csv_path.exists():
        # Fallback 
        csv_path = EXTRACTED_DIR / "annotations" / "annotations_train.csv"
        if not csv_path.exists():
            print(f"Error: Couldn't find annotations CSV in {EXTRACTED_DIR}. Did extraction finish?")
            return
            
    print(f"Loading annotations from {csv_path}...")
    df = pd.read_csv(csv_path, header=None)
    
    # Ensure correct columns based on standard SKU110K
    if len(df.columns) >= 8:
        df.columns = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class_id', 'image_width', 'image_height'] + list(df.columns[8:])
    else:
        print("Format error.")
        return
        
    unique_images = df['image_name'].unique()
    
    # Split 80 / 10 / 10
    print("\nSplitting dataset (80% Train, 10% Val, 10% Test)...")
    train_imgs, temp_imgs = train_test_split(unique_images, test_size=0.20, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)
    
    print(f"Train size: {len(train_imgs)} images")
    print(f"Val size:   {len(val_imgs)} images")
    print(f"Test size:  {len(test_imgs)} images")
    
    create_yolo_dirs()
    
    # This process handles the physical copying
    process_and_copy(df, train_imgs, 'train')
    process_and_copy(df, val_imgs, 'val')
    process_and_copy(df, test_imgs, 'test')
    
    print("\n✅ YOLO Dataset completely prepared and deployed onto drive!")

if __name__ == "__main__":
    run()
