# Retail Shelf Monitoring via Object Detection
# Main CLI Orchestrator

import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Retail Shelf Monitoring Pipeline Engine")
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['prepare', 'visualize', 'train_yolo', 'train_resnet', 'evaluate', 'inference', 'dashboard'],
                        help="Select which phase of the project to run.")
                        
    args = parser.parse_args()

    if args.mode == 'prepare':
        print("🚀 Booting Data Preparation Pipeline...")
        try:
            # Since main.py is inside the src folder, we can just import it directly
            import prepare_yolo_dataset
            prepare_yolo_dataset.run()
        except ImportError as e:
            print(f"Error loading preparation module: {e}")
            
    elif args.mode == 'visualize':
        print("🚀 Booting Custom PyTorch Dataset Verification...")
        try:
            import visualize_augmentation
            visualize_augmentation.run()
        except ImportError as e:
            print(f"Error loading visualization module: {e}")
            
    elif args.mode == 'train_yolo':
        print("🚀 Booting YOLO Model Training pipeline...")
        try:
            import train_yolo
            train_yolo.run()
        except ImportError as e:
            print(f"Error loading training module: {e}")
            
    elif args.mode == 'train_resnet':
        print("🚀 Booting ResNet-50 Training pipeline...")
        try:
            import train_resnet
            train_resnet.run()
        except ImportError as e:
            print(f"Error loading training module: {e}")
        
    elif args.mode == 'evaluate':
        print("🚀 Booting Evaluation...")
        print("Evaluation mode ready to be implemented.")
        
    elif args.mode == 'inference':
        print("🚀 Booting Inference & Business Logic pipeline...")
        try:
            import inference
            inference.run()
        except ImportError as e:
            print(f"Error loading inference module: {e}")
            
    elif args.mode == 'dashboard':
        print("🚀 Booting OpenCV Visualisation & Dashboard generation...")
        try:
            import visualize_results
            visualize_results.run()
        except ImportError as e:
            print(f"Error loading visualization module: {e}")

if __name__ == "__main__":
    main()
