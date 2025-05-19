import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

def extract_zip_with_cleanup(root_path, extract_to=None, delete_after=False):
    """
    Распаковывает zip-архив и опционально удаляет его после распаковки

    :param zip_path: Путь к zip-архиву
    :param extract_to: Папка для распаковки (по умолчанию - рядом с архивом)
    :param delete_after: Удалять ли архив после распаковки
    """
    zip_path = os.path.join(root_path, "images.zip")
    images_fake_path = os.path.join(root_path, "fake")
    images_real_path = os.path.join(root_path, "real")
    if os.path.exists(images_fake_path) or os.path.exists(images_real_path):
        return images_fake_path, images_real_path
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    os.makedirs(extract_to, exist_ok=True)

    print(f"Extracting {zip_path} в {extract_to}...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            try:
                zip_ref.extract(file, extract_to)
            except FileExistsError:
                print(f"File {file} already exist, skip")

    print(f"Finish. Extructed {len(zip_ref.namelist())} files.")

    if delete_after:
        print(f"Delete archive {zip_path}...")
        os.remove(zip_path)
        print("Archive deleted.")
    return images_fake_path, images_real_path

def prepare_and_save_data(
    real_dir, 
    fake_dir, 
    output_dir="data/dataset",
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    max_samples=None,
    target_size=(299, 299)
):
    """
    Подготавливает и сохраняет данные в структурированные директории
    
    Parameters:
        output_dir (str): Корневая директория для сохранения
        target_size (tuple): Размер изображений (width, height)
    """

    splits = ['train', 'val', 'test']
    classes = ['real', 'fake']
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    if os.path.exists(output_dir):
        return train_dir, val_dir, test_dir
    
    for split in splits:
        for cls in classes:
            create_directory(os.path.join(output_dir, split, cls))
    
    real_images = [f for f in os.listdir(real_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    fake_images = [f for f in os.listdir(fake_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if max_samples:
        real_images = real_images[:max_samples]
        fake_images = fake_images[:max_samples]
    
    real_train, real_temp = train_test_split(
        real_images, 
        test_size=val_size + test_size, 
        random_state=random_state
    )
    real_val, real_test = train_test_split(
        real_temp,
        test_size=test_size / (val_size + test_size),
        random_state=random_state
    )
    
    fake_train, fake_temp = train_test_split(
        fake_images,
        test_size=val_size + test_size,
        random_state=random_state
    )
    fake_val, fake_test = train_test_split(
        fake_temp,
        test_size=test_size / (val_size + test_size),
        random_state=random_state
    )
    
    def save_images(image_list, source_dir, target_dir):
        for img_name in image_list:
            img_path = os.path.join(source_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            cv2.imwrite(os.path.join(target_dir, img_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    save_images(real_train, real_dir, os.path.join(output_dir, 'train', 'real'))
    save_images(real_val, real_dir, os.path.join(output_dir, 'val', 'real'))
    save_images(real_test, real_dir, os.path.join(output_dir, 'test', 'real'))
    
    save_images(fake_train, fake_dir, os.path.join(output_dir, 'train', 'fake'))
    save_images(fake_val, fake_dir, os.path.join(output_dir, 'val', 'fake'))
    save_images(fake_test, fake_dir, os.path.join(output_dir, 'test', 'fake'))
    
    print("Data is prepared:", output_dir)
    print(f"Train: {len(real_train)} real, {len(fake_train)} fake")
    print(f"Val: {len(real_val)} real, {len(fake_val)} fake")
    print(f"Test: {len(real_test)} real, {len(fake_test)} fake")
    
    return train_dir, val_dir, test_dir


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)