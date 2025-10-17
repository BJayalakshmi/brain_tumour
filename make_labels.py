import os

train_img = "data/images/train"
val_img = "data/images/val"
train_label = "data/labels/train"
val_label = "data/labels/val"

os.makedirs(train_label, exist_ok=True)
os.makedirs(val_label, exist_ok=True)

def create_labels(img_dir, label_dir):
    for img_name in os.listdir(img_dir):
        cls = 0 if "yes" in img_name else 1
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        # YOLO format: class x_center y_center width height (normalized)
        with open(label_path, "w") as f:
            f.write(f"{cls} 0.5 0.5 1.0 1.0\n")

create_labels(train_img, train_label)
create_labels(val_img, val_label)

print("✅ Labels created!")
