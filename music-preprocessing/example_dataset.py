import os
import shutil
from pathlib import Path

'''
Script that makes example (dummy) dataset for presentation/github that shows example of file hierarchy if 
data_processing scripts are ran.
'''

SOURCE_DIR = "data_opt/large_mel_images"
DEST_DIR = "example_dataset/gray"

# Ensure destination exists
Path(DEST_DIR).mkdir(parents=True, exist_ok=True)

for genre_dir in os.listdir(SOURCE_DIR):
    full_genre_path = os.path.join(SOURCE_DIR, genre_dir)
    if os.path.isdir(full_genre_path):
        dest_genre_path = os.path.join(DEST_DIR, genre_dir)
        os.makedirs(dest_genre_path, exist_ok=True)

        # Find first PNG file in the genre folder
        for file in os.listdir(full_genre_path):
            if file.endswith(".png"):
                src = os.path.join(full_genre_path, file)
                dst = os.path.join(dest_genre_path, file)
                shutil.copy2(src, dst)
                break
