import os
import json
import glob
import random
import shutil
import cv2
from dataclasses import dataclass, asdict
from typing import List, Tuple
from ultralytics import YOLO

@dataclass
class AnalysisReport:
    image_path: str
    total_products_detected: int
    out_of_stock_zones: List[dict]
    planogram_compliance: float
    planogram_violations: List[dict]

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    box = [x_min, y_min, x_max, y_max]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def detect_out_of_stock(boxes, img_width, img_height, grid_rows=3, grid_cols=4):
    """
    Divide the image into a grid. If a grid cell has no bounding box center
    inside it, flag it as an Out-of-Stock (OOS) zone.
    """
    oos_zones = []
    cell_w = img_width / grid_cols
    cell_h = img_height / grid_rows

    # Initialize grid counts
    grid_counts = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]

    for box in boxes:
        # box is [x_min, y_min, x_max, y_max]
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        
        col = int(cx / cell_w)
        row = int(cy / cell_h)
        
        if 0 <= row < grid_rows and 0 <= col < grid_cols:
            grid_counts[row][col] += 1

    for row in range(grid_rows):
        for col in range(grid_cols):
            if grid_counts[row][col] == 0:
                # OOS recognized
                oos_zones.append({
                    "grid_coord": [row, col],
                    "region": [col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h]
                })

    return oos_zones

def verify_planogram(detected_boxes, planogram_path, img_width, img_height, iou_threshold=0.15):
    """
    Compare detected boxes vs expected planogram boxes.
    If an expected box doesn't have an IoU overlap > iou_threshold with any
    detected box, it is a planogram violation.
    """
    violations = []
    try:
        with open(planogram_path, 'r') as f:
            planogram = json.load(f)
    except Exception as e:
        print(f"Failed to load planogram JSON: {e}")
        return 0.0, []

    expected_products = planogram.get("expected_products", [])
    if not expected_products:
        return 100.0, []

    matched_expected = 0
    
    for slot in expected_products:
        # relative box to absolute max coords
        r_box = slot["bbox_relative"]
        expected_box = [r_box[0] * img_width, r_box[1] * img_height, 
                        r_box[2] * img_width, r_box[3] * img_height]
        
        slot_matched = False
        for d_box in detected_boxes:
            iou = calculate_iou(expected_box, d_box)
            if iou > iou_threshold:
                slot_matched = True
                break
        
        if slot_matched:
            matched_expected += 1
        else:
            violations.append({
                "expected_id": slot.get("id", "unknown"),
                "expected_box": expected_box,
                "issue": "Missing product or misalignment"
            })

    compliance_rate = (matched_expected / len(expected_products)) * 100
    return compliance_rate, violations


def run_inference(image_paths, model_path, output_dir, planogram_path):
    print(f"Loading YOLO model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    all_reports = []

    for img_path in image_paths:
        print(f"Processing {img_path}...")
        results = model.predict(source=img_path, conf=0.25, save=False, verbose=False)
        result = results[0]
        
        # Original image dimensions
        img_height, img_width = result.orig_shape
        
        # Get detected bounding boxes
        boxes = result.boxes.xyxy.cpu().numpy().tolist()
        num_products = len(boxes)

        #  OOS Logic
        oos_zones = detect_out_of_stock(boxes, img_width, img_height)

        # Planogram Logic
        compliance_rate, violations = verify_planogram(boxes, planogram_path, img_width, img_height)

        # Create Report
        report = AnalysisReport(
            image_path=img_path,
            total_products_detected=num_products,
            out_of_stock_zones=oos_zones,
            planogram_compliance=compliance_rate,
            planogram_violations=violations
        )
        
        report_dict = asdict(report)
        all_reports.append(report_dict)

        
        base_name = os.path.basename(img_path)
        json_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_report.json")
        with open(json_path, 'w') as f:
            json.dump(report_dict, f, indent=4)
            
        
        dest_img_path = os.path.join(output_dir, f"original_{base_name}")
        shutil.copy(img_path, dest_img_path)
        
        
        pred_img_path = os.path.join(output_dir, f"model_predictions_{base_name}")
        result.save(filename=pred_img_path)
        
        
        label_file_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        gt_img_path = os.path.join(output_dir, f"human_annotations_{base_name}")
        
        if os.path.exists(label_file_path):
            img_gt = cv2.imread(img_path)
            with open(label_file_path, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class_id, center_x, center_y, width, height (normalized)
                        cx = float(parts[1]) * img_width
                        cy = float(parts[2]) * img_height
                        w = float(parts[3]) * img_width
                        h = float(parts[4]) * img_height
                        
                        x1 = int(cx - w / 2)
                        y1 = int(cy - h / 2)
                        x2 = int(cx + w / 2)
                        y2 = int(cy + h / 2)
                        
                        # Draw green box for human annotation
                        cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(gt_img_path, img_gt)
            
        print(f"Report, predictions, and ground truth logic saved for {base_name}")
    
    print(f"Successfully processed {len(image_paths)} images.")

def run():
    # Setup paths
    model_path = os.path.join("runs", "detect", "outputs", "yolo_unfrozen", "weights", "best.pt")
    data_dir = os.path.join("data", "images", "test")
    planogram_path = os.path.join("data", "sample_planogram.json")
    output_dir = os.path.join("outputs", "reports")
    
    # ---------------------------------------------------------
    # Note 
    # To use a specific set of images instead of random selection the code is 
    # 
    # Example:
    # specific_images = [
    #     r"data\images\test\shelf_2.jpg"
    # ]
    # run_inference(specific_images, model_path, output_dir, planogram_path)
    # return
    # ---------------------------------------------------------

    # get random test images
    if not os.path.exists(data_dir):
        print(f"Test directory not found: {data_dir}")
        return
        
    all_images = glob.glob(os.path.join(data_dir, "*.jpg"))
    if not all_images:
        print("No .jpg images found in the test directory.")
        return
        
    num_to_sample = min(10, len(all_images))
    sample_images = random.sample(all_images, num_to_sample)
    
    run_inference(sample_images, model_path, output_dir, planogram_path)


if __name__ == "__main__":
    run()
