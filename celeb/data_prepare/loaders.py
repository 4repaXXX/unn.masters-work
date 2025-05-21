import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_prepare.dataset_tools import extract_zip_with_cleanup, prepare_and_save_data, create_directory
from data_prepare.models import ModelType

class DataLoader:
    def __init__(self, image_archive_path, output_dir, preprocess_func, image_size):
        self._image_archive_path = image_archive_path
        self._augmentation = augmentation
        self._preprocess_func = preprocess_func
        self._output_dir = output_dir
        self._image_size = image_size
            
        create_directory(self._output_dir)

    def load(self):
        return self._init_data()

    def load_genrators(self):
        train_dir, val_dir, test_dir = self._init_data(self._output_dir)
        print(train_dir, val_dir, test_dir)
        return self._create_data_generators(train_dir, val_dir, test_dir)

    def _init_data(self, output_dir):
        fake_images_path, real_images_path = extract_zip_with_cleanup(self._image_archive_path)
        print(fake_images_path, real_images_path)
        train_dir, val_dir, test_dir = prepare_and_save_data(
            real_images_path, 
            fake_images_path, 
            output_dir
        )
        return train_dir, val_dir, test_dir

    def _create_data_generators(
        self,
        train_dir, 
        val_dir, 
        test_dir,
        batch_size=32,
        random_state=42,
        class_mode='binary'
    ):
        """
        Создает потоковые генераторы для train/val/test данных
    
        Parameters:
            data_dir (str): Путь к директории с данными (должна содержать train/val/test)
            target_size (tuple): Размер изображений (ширина, высота)
            batch_size (int): Размер батча
            augmentation (bool): Применять ли аугментацию для тренировочных данных
            class_mode (str): Тип классификации ('binary' или 'categorical')
    
        Returns:
            tuple: (train_gen, val_gen, test_gen)
        """
        
        train_datagen = ImageDataGenerator(
            preprocessing_function=self._preprocess_func
        )
        
        val_test_datagen = ImageDataGenerator(
            preprocessing_function=self._preprocess_func
        )
        
        # Labels
        # real/ -> class 0
        # fake/ -> class 1
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=self._image_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=True,
            seed=random_state
        )
        
        val_gen = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self._image_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False
        )
        
        test_gen = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self._image_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False
        )
        
        print("\nClass indices:", train_gen.class_indices)
        print(f"Train samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
        
        return train_gen, val_gen, test_gen

class XDataLoader(DataLoader):
    def __init__(self, image_archive_path, output_dir="data/dataset"):
        output_dir = os.path.join(output_dir, 'x')
        image_size = (299, 299)
        super().__init__(image_archive_path, output_dir, tf.keras.applications.xception.preprocess_input, image_size)

class NetDataLoader(DataLoader):
    def __init__(self, image_archive_path, output_dir="data/dataset"):
        image_size = (224, 224)
        output_dir = os.path.join(output_dir, 'net')
        super().__init__(image_archive_path, output_dir, tf.keras.applications.efficientnet.preprocess_input, image_size)