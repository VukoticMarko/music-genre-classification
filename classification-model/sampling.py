import os
import random
import shutil
from sklearn.model_selection import train_test_split

CLASS_LABELS = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'easy listening': 3,
    'electronic': 4,
    'experimental': 5,
    'folk': 6,
    'hip-hop': 7,
    'instrumental': 8,
    'international': 9,
    'jazz': 10,
    'pop': 11,
    'rock': 12,
    'soul-rnb': 13,
    'spoken': 14
}

CLASS_LABELS = { # This is used when we want to use less classes than available
    'electronic': 0,
    'experimental': 1,
    'folk': 2,
    'hip-hop': 3,
    'pop': 4,
    'rock': 5,
}

def sample_dataset(
    source_dir,
    dest_dir,
    max_images_per_class=200,
    val_split=0.15,
    test_split=0.10,
    delete_after=False
):
    """
    Randomly samples a subset of images from each allowed class, splits into train/val/test,
    and copies them into a new directory structure.

    Args:
        source_dir (str): Path to the original dataset directory.
        dest_dir (str): Path where the sampled dataset will be stored.
        max_images_per_class (int): Max number of images to keep per class.
        val_split (float): Fraction of images to use for validation.
        test_split (float): Fraction of images to use for testing.
        delete_after (bool): Whether to delete the sampled directory after use.

    Returns:
        str: Path to the sampled dataset (dest_dir).
    """
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Sampling dataset from: {source_dir}")
    print(f"Saving temporary dataset to: {dest_dir}")

    allowed_classes = set(CLASS_LABELS.keys())

    for class_name in os.listdir(source_dir):
        if class_name not in allowed_classes:
            print(f"⏭Skipping unlisted class: {class_name}")
            continue

        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not images:
            continue

        # Randomly sample
        random.shuffle(images)
        images = images[:max_images_per_class]

        # Split into train/val/test
        temp_train, test = train_test_split(images, test_size=test_split, random_state=42)
        train, val = train_test_split(temp_train, test_size=val_split / (1 - test_split), random_state=42)

        # Copy into new structure
        for subset, subset_images in [('train', train), ('val', val), ('test', test)]:
            dest_subset_dir = os.path.join(dest_dir, subset, class_name)
            os.makedirs(dest_subset_dir, exist_ok=True)
            for img_path in subset_images:
                shutil.copy(img_path, dest_subset_dir)

    print("Sampling completed.")
    return dest_dir


def delete_sampled_dataset(path):
    """Deletes the sampled dataset directory."""
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted temporary sampled dataset: {path}")