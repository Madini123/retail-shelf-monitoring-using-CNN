import os
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil

def draw_glass_panel(img, text_lines):
    
    overlay = img.copy()
    panel_w = 550
    panel_h = 30 + 50 * len(text_lines)
    
    cv2.rectangle(overlay, (30, 30), (30 + panel_w, 30 + panel_h), (0, 0, 0), -1)
    
    alpha = 0.55  
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    cv2.rectangle(img, (30, 30), (30 + panel_w, 30 + panel_h), (255, 255, 255), 2)
    
    y = 80
    for i, line in enumerate(text_lines):
        
        thickness = 3 if i == 0 else 2
        font_scale = 1.2 if i == 0 else 1.0
        color = (200, 255, 255) if i == 0 else (255, 255, 255) 
        
        cv2.putText(img, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y += 50
        
    return img

def process_image(img_path, json_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    with open(json_path, 'r') as f:
        report = json.load(f)
        
   
    overlay = img.copy()
    for oos in report.get("out_of_stock_zones", []):
        r = oos["region"]
        x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        
    
    alpha = 0.35
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    
    for vio in report.get("planogram_violations", []):
        r = vio["expected_box"]
        x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 5)
        
        cv2.putText(img, "PLANOGRAM VIOLATION", (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(img, "PLANOGRAM VIOLATION", (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
        
    
    text = [
        "RETAIL SHELF ANALYSIS",
        f"Total Products: {report.get('total_products_detected', 0)}",
        f"OOS Zones: {len(report.get('out_of_stock_zones', []))}",
        f"Planogram Compliance: {report.get('planogram_compliance', 0.0):.1f}%"
    ]
    img = draw_glass_panel(img, text)
    
    cv2.imwrite(output_path, img)
    return img

def create_matplotlib_grid(image_paths, output_grid_path):
    
    n = len(image_paths)
    if n == 0:
        return
        
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    if rows > 3:
        rows = 3
        
    n = min(n, 9)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(len(axes)):
        if i < n:
            img = cv2.imread(image_paths[i])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].axis('off')
            axes[i].set_title(f"Test Feed {i+1}", fontsize=14, fontweight='bold')
        else:
            axes[i].axis('off')
            
    plt.tight_layout()
    plt.savefig(output_grid_path, dpi=200, bbox_inches='tight')
    plt.close()

def create_demo_video(image_pairs, output_video_path, fps=1):
    
    if not image_pairs:
        return
        
    
    first_img = cv2.imread(image_pairs[0][1])
    h, w, _ = first_img.shape
    
    # Scale down drastically for Web/GitHub limits (max width 800)
    scale = 800 / float(w) if w > 800 else 1.0
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_w, new_h))
    
    for orig_path, dash_path in image_pairs:
        # Show original frame for 1 second
        orig_img = cv2.imread(orig_path)
        if orig_img is not None:
            orig_img = cv2.resize(orig_img, (new_w, new_h))
            out.write(orig_img)
            
        # Show dashboard result frame for 1 second
        dash_img = cv2.imread(dash_path)
        if dash_img is not None:
            dash_img = cv2.resize(dash_img, (new_w, new_h)) 
            out.write(dash_img)
        
    out.release()

def run():
    report_dir = os.path.join("outputs", "reports")
    assets_dir = os.path.join("assets")
    vis_out_dir = os.path.join(assets_dir, "visualized_frames")
    
    
    if os.path.exists(vis_out_dir):
        shutil.rmtree(vis_out_dir)
    os.makedirs(vis_out_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(report_dir, "*_report.json"))
    processed_dashboards = []
    video_pairs = []
    
    for json_path in json_files:
        base_name = os.path.basename(json_path).replace("_report.json", ".jpg")
        
        orig_path = os.path.join(report_dir, f"original_{base_name}")
        img_path = os.path.join(report_dir, f"model_predictions_{base_name}")
        
        if not os.path.exists(img_path):
            continue
            
        out_path = os.path.join(vis_out_dir, f"dashboard_{base_name}")
        img = process_image(img_path, json_path, out_path)
        
        if img is not None:
            processed_dashboards.append(out_path)
            if os.path.exists(orig_path):
                video_pairs.append((orig_path, out_path))
            print(f"Generated dashboard for {base_name}")
            
    print("\nGenerating Matplotlib results grid...")
    grid_path = os.path.join(assets_dir, "results_grid.png")
    create_matplotlib_grid(processed_dashboards, grid_path)
    
    print("Generating MP4 Demo Video...")
    video_path = os.path.join(assets_dir, "demo_video.mp4")
    create_demo_video(video_pairs, video_path, fps=1)
    
    print(f"\n✅ Visual assets successfully saved to '{assets_dir}/'")

if __name__ == "__main__":
    run()
