from pathlib import Path
import shutil
import random

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / 'archive' / 'person-3'
TRAIN_DIR = DATASET_ROOT / 'train'
VALID_DIR = DATASET_ROOT / 'valid'
TEST_DIR = DATASET_ROOT / 'test'


def ensure_split():
    image_dir = TRAIN_DIR / 'images'
    label_dir = TRAIN_DIR / 'labels'
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f'Missing training folders: {image_dir} or {label_dir}')

    img_files = sorted(image_dir.glob('*'))
    # Only include files that have a matching label. Skip invalid entries.
    valid_img_files = []
    for img in img_files:
        if img.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        label_file = label_dir / (img.stem + '.txt')
        if label_file.exists():
            valid_img_files.append(img)

    if not valid_img_files:
        raise FileNotFoundError(f'No valid image/label pairs found in {image_dir}')

    val_images_dir = VALID_DIR / 'images'
    val_labels_dir = VALID_DIR / 'labels'
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    (TEST_DIR / 'images').mkdir(parents=True, exist_ok=True)
    (TEST_DIR / 'labels').mkdir(parents=True, exist_ok=True)

    # Keep a fixed split so results are reproducible.
    random.seed(42)
    val_count = max(1, round(len(valid_img_files) * 0.1))
    val_files = random.sample(valid_img_files, val_count)
    val_set = {f.name for f in val_files}

    for img in valid_img_files:
        label_file = label_dir / (img.stem + '.txt')
        if img.name in val_set:
            shutil.copy2(img, val_images_dir / img.name)
            shutil.copy2(label_file, val_labels_dir / label_file.name)
            img.unlink()
            label_file.unlink()

    train_total = len(list(image_dir.glob('*')))
    return train_total, val_count


def write_data_yaml():
    data_yaml = DATASET_ROOT / 'data.yaml'
    data_yaml.write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n\n"
        "nc: 1\n"
        "names: ['object']\n",
        encoding='utf-8',
    )
    return data_yaml


def main():
    train_total, val_total = ensure_split()
    data_yaml = write_data_yaml()
    print(f'Train dataset images: {train_total}')
    print(f'Validation dataset images: {val_total}')
    print(f'Using config: {data_yaml}')

    model = YOLO(ROOT / 'yolov8n.pt')
    results = model.train(
        data=str(data_yaml),
        epochs=10,
        imgsz=640,
        batch=16,
        project=str(ROOT / 'runs' / 'train'),
        name='person_detection',
        exist_ok=True,
        verbose=True,
    )
    print('Training finished.')
    print(results)


if __name__ == '__main__':
    main()
