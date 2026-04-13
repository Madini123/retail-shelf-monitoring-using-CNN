import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ShelfDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df (pandas.DataFrame): DataFrame containing annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample (Albumentations).
        """
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # We need a list of unique images
        self.image_names = df['image_name'].unique()
        
        self.grouped_df = df.groupby('image_name')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_names[idx]
        img_path = self.img_dir / img_name
        
        # Handle extension issues
        if not img_path.exists():
            img_path = self.img_dir / f"{img_name}.jpg"

        # Load Image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        h_img, w_img = image.shape[:2]

        # Get bounding boxes: format [x_center, y_center, width, height, class_name]
        group = self.grouped_df.get_group(img_name)
        bboxes = []
        for _, row in group.iterrows():
            
            x1 = max(0.0, min(float(row['x1']), float(w_img)))
            y1 = max(0.0, min(float(row['y1']), float(h_img)))
            x2 = max(0.0, min(float(row['x2']), float(w_img)))
            y2 = max(0.0, min(float(row['y2']), float(h_img)))
            
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            x_center = ((x1 + x2) / 2) / w_img
            y_center = ((y1 + y2) / 2) / h_img
            w = (x2 - x1) / w_img
            h = (y2 - y1) / h_img
            
            
            x_center = np.clip(x_center, 0.0, 1.0)
            y_center = np.clip(y_center, 0.0, 1.0)
            w = np.clip(w, 0.0, 1.0)
            h = np.clip(h, 0.0, 1.0)
            
            
            bboxes.append([x_center, y_center, w, h, 0])

        if self.transform:
            try:
                transformed = self.transform(image=image, bboxes=bboxes)
                image = transformed['image']
                bboxes = transformed['bboxes']
            except ValueError as e:
                print(f"Albumentations Error on {img_name}: {e}. Falling back...")
                # If albumentations fails, Normalize or else the Denormalization math washes out the photo!
                import albumentations as fallback_A
                from albumentations.pytorch import ToTensorV2
                fallback_transform = fallback_A.Compose([
                    fallback_A.Resize(640, 640), 
                    fallback_A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
                transformed = fallback_transform(image=image)
                image = transformed['image']
                bboxes = []

        
        
        return image, bboxes

def get_train_transforms():
    """ Albumentations pipeline focusing heavily on dense shelf detection challenges. """
    return A.Compose(
        [
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ],
        
        bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=[])
    )

def custom_collate_fn(batch):
    """
    Since each image has a different number of products on the shelf,
    we cannot 'stack' bounding boxes into a perfect rectangle tensor.
    This safely packages the variable length boxes.
    """
    images = []
    bboxes = []
    for img, bbox in batch:
        images.append(img)
        bboxes.append(bbox)
        
    images = torch.stack(images, dim=0)
    return images, bboxes
    
def test_loader(df_path, img_dir):
    """ Demo function to verify the DataLoader runs perfectly. """
    print("Initializing Custom ShelfDataset...")
    df = pd.read_csv(df_path, header=None)
    
    df.columns = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class_id', 'image_width', 'image_height'] + list(df.columns[8:])
    
    transforms = get_train_transforms()
    dataset = ShelfDataset(df=df, img_dir=img_dir, transform=transforms)
    
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    
    print("Testing loader retrieval...")
    batch_idx, (images, bboxes) = next(enumerate(loader))
    print(f"Batch loaded. Image tensor shape: {images.shape} (Batch, Channels, Height, Width)")
    print("Data loading system is complete.")
    
    return loader
