from ultralytics import YOLO
from pathlib import Path
import torch

def run():
    print("====================================")
    print(" YOLOv8 Baseline Training Engine")
    print("====================================\n")
    
    
    if not torch.cuda.is_available():
        print("\n[WARNING] No CUDA GPU detected! Training this model for 50 epochs on a CPU will take an extraordinary amount of time.")
        print("We highly recommend compressing this folder and moving to Google Colab, or pressing Ctrl+C right now if you are just testing.\n")
    else:
        print(f"✅ Utilizing GPU: {torch.cuda.get_device_name(0)}\n")

    
    model = YOLO("yolov8m.pt")
    
    
    yaml_path = Path("data/data.yaml").absolute()
    
    if not yaml_path.exists():
        print(f"Error: {yaml_path} does not exist. Ensure data.yaml was created properly in the data folder.")
        return

    print("--- STAGE 1: Frozen Backbone Training (30 Epochs) ---")
    print("✅ Stage 1 already completed! Skipping to Stage 2...")
    # freeze=10 locks the first 10 core neural network layers. 
    # This mitigates 'catastrophic forgetting' so it keeps general object knowledge.
    model.train(
        data=str(yaml_path),
        epochs=30,
        imgsz=640,
        batch=4,
        freeze=10,
        amp=False,
        workers=0,
        device=0,
        project="outputs",
        name="yolo_frozen"
    )
    
    print("\n--- STAGE 2: Full Fine-Tuning (20 Epochs) ---")
    
    best_frozen = Path("runs/detect/outputs/yolo_frozen/weights/best.pt")
    if best_frozen.exists():
        model_unfrozen = YOLO(str(best_frozen))
    else:
        print(f"Could not find weights from State 1 at {best_frozen}. Fine tuning cannot proceed.")
        return
        
    
    model_unfrozen.train(
        data=str(yaml_path),
        epochs=20,
        imgsz=640,
        batch=2,
        freeze=0,
        lr0=1e-4, 
        amp=False,
        workers=0,
        device=0,
        project="outputs",
        name="yolo_unfrozen"
    )
    
    print("\n✅ YOLOv8 Pipeline Complete! Test visualizations and weights are stored in outputs/yolo_unfrozen/")
