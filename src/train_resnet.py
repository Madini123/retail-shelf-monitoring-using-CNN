import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import cv2
import numpy as np
from pathlib import Path

class ResNetShelfDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.image_names = df['image_name'].unique()
        self.grouped_df = df.groupby('image_name')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.img_dir / img_name
        if not img_path.exists():
            img_path = self.img_dir / f"{img_name}.jpg"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img = image.shape[:2]

        group = self.grouped_df.get_group(img_name)
        bboxes = []
        labels = []
        for _, row in group.iterrows():
            x1 = max(0.0, min(float(row['x1']), float(w_img)))
            y1 = max(0.0, min(float(row['y1']), float(h_img)))
            x2 = max(0.0, min(float(row['x2']), float(w_img)))
            y2 = max(0.0, min(float(row['y2']), float(h_img)))

            if x2 <= x1 or y2 <= y1:
                continue
                
            
            bboxes.append([x1, y1, x2, y2])
            labels.append(1) 

        if self.transform and len(bboxes) > 0:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        elif self.transform:
            transformed = self.transform(image=image, bboxes=[], labels=[])
            image = transformed['image']
            
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes if len(bboxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        return image, target

def get_resnet_transform():
    return A.Compose([
        # Resize to 640x640 since YOLO model uses that
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def custom_collate(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    # Load model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def run():
    print("=======================================")
    print(" Faster R-CNN (ResNet-50) Training Node")
    print("=======================================\n")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"✅ Utilizing compute device: {device}\n")
    if device.type == 'cpu':
        print("[WARNING] You are training Faster R-CNN on a CPU locally. This will be slow, but proceeding as requested!")

    # Paths
    TRAIN_CSV = Path("data/SKU110K_fixed/annotations/annotations_train.csv")
    IMG_DIR = Path("data/SKU110K_fixed/images")
    
    # Simple fallback mapping if files are strictly packed
    if not TRAIN_CSV.exists():
        print(f"Could not find {TRAIN_CSV}. Make sure you pass the correct annotations and image directory path.")
        
    # Initialization
    df = pd.read_csv(TRAIN_CSV, header=None)
    df.columns = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class_id', 'image_width', 'image_height']
    dataset = ResNetShelfDataset(df, IMG_DIR, get_resnet_transform())
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=custom_collate)
    
    model = get_model(num_classes=2) 
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    epochs = 15
    print(f"Beginning local training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i}/{len(data_loader)}] Loss: {losses.item():.4f}")
            
        print(f"--- Epoch: {epoch+1} Average Loss: {epoch_loss/len(data_loader):.4f} ---")
        
        
        model_save_path = Path(f"models/resnet50_epoch_{epoch+1}.pth")
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
    
    print("✅ ResNet-50 Local Training Script completed successfully!")

if __name__ == "__main__":
    run()
