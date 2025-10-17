import os, shutil, random

# Paths
base = "data"
yes_dir = os.path.join(base, "yes")
no_dir = os.path.join(base, "no")

train_img = os.path.join(base, "images/train")
val_img = os.path.join(base, "images/val")

os.makedirs(train_img, exist_ok=True)
os.makedirs(val_img, exist_ok=True)

split_ratio = 0.8  # 80% train, 20% validation

for label_dir, label in [(yes_dir, "yes"), (no_dir, "no")]:
    files = [f for f in os.listdir(label_dir) if f.endswith(".jpg") or f.endswith(".png")]
    random.shuffle(files)
    split_idx = int(len(files) * split_ratio)
    for i, file in enumerate(files):
        src = os.path.join(label_dir, file)
        if i < split_idx:
            dst = os.path.join(train_img, f"{label}_{file}")
        else:
            dst = os.path.join(val_img, f"{label}_{file}")
        shutil.copy(src, dst)

print("✅ Data split complete!")
