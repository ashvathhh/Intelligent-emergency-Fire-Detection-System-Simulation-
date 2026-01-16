import json
import os

# Define paths
json_path = 'C:/Users/shbha/OneDrive/Documents/Semester Project/data/labels/train.json'  # Path to the JSON file
output_dir = 'C:/Users/shbha/OneDrive/Documents/Semester Project/data/labels/yolo_labels'  # Directory to save YOLO labels

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to convert COCO bbox to YOLO format
def convert_bbox(size, box):
    x, y, width, height = box
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (x + width / 2.0) * dw
    y_center = (y + height / 2.0) * dh
    width *= dw
    height *= dh
    return x_center, y_center, width, height

# Load JSON data
with open(json_path, 'r') as f:
    data = json.load(f)

# Mapping image IDs to their width and height
images = {img["id"]: img for img in data["images"]}

# Process annotations
for ann in data["annotations"]:
    image_info = images[ann["image_id"]]
    image_width, image_height = image_info["width"], image_info["height"]
    bbox = ann["bbox"]
    
    # Convert bbox to YOLO format
    yolo_bbox = convert_bbox((image_width, image_height), bbox)
    
    # Prepare YOLO label line
    category_id = ann["category_id"]  # Adjust if you have custom class mappings
    label_line = f"{category_id} " + " ".join(map(str, yolo_bbox)) + "\n"
    
    # Write to .txt file named after the image
    txt_filename = os.path.join(output_dir, f"{image_info['file_name'].split('.')[0]}.txt")
    with open(txt_filename, "a") as label_file:
        label_file.write(label_line)

print("Conversion complete. YOLO label files are in:", output_dir)
