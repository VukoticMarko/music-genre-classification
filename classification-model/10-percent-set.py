import os
import numpy as np
import shutil
data_dir = '../music-preprocessing/data_opt/large_mel_images'

# --- Step 1: Get all class subfolders ---
all_classes = sorted(os.listdir(data_dir))

# --- Step 2: Reduce each class to 10% of its files ---
def make_subset_folder(original_dir, subset_dir, fraction=0.1):
    os.makedirs(subset_dir, exist_ok=True)
    for cls in sorted(os.listdir(original_dir)):
        cls_src = os.path.join(original_dir, cls)
        if not os.path.isdir(cls_src):
            continue
        cls_dst = os.path.join(subset_dir, cls.replace(" ", "_"))  # safer folder names
        os.makedirs(cls_dst, exist_ok=True)

        all_files = [f for f in os.listdir(cls_src) if os.path.isfile(os.path.join(cls_src, f))]
        np.random.shuffle(all_files)
        n_keep = max(1, int(len(all_files) * fraction))
        keep_files = all_files[:n_keep]

        for f in keep_files:
            src_path = os.path.join(cls_src, f)
            dst_path = os.path.join(cls_dst, f)
            shutil.copy2(src_path, dst_path)  # copy instead of link

subset_dir = "../music-preprocessing/data_subset_10p"

make_subset_folder(data_dir, subset_dir, fraction=0.1)
print(f"✅ Created 10% subset at: {subset_dir}")
