import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import Xception, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPool2D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from enum import Enum
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_prepare.dataset_tools import extract_zip_with_cleanup, prepare_and_save_data, create_directory

class ModelType(Enum):
    XCEPTION = "xception"
    EFFICIENTNET = "efficientnet"
    CUSTOM = "custom_cnn"

class DataLoader:
    def __init__(self, image_archive_path, model_type=ModelType.XCEPTION, augmentation=False):
        self._image_archive_path = image_archive_path
        self._model_type = model_type
        self._augmentation = augmentation
        self._preprocess_func = tf.keras.applications.xception.preprocess_input
        self._postfix = 'x'
        if self._model_type == ModelType.EFFICIENTNET:
            self._postfix = 'net'
            self._preprocess_func = tf.keras.applications.efficientnet.preprocess_input
            

    def load(self, output_dir):
        return self._init_data(os.path.join(output_dir, self._postfix))

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
        target_size=(299, 299),
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
        
        if self._augmentation:
            train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_func,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
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
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=True,
            seed=random_state
        )
        
        val_gen = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False
        )
        
        test_gen = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False
        )
        
        print("\nClass indices:", train_gen.class_indices)
        print(f"Train samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
        
        return train_gen, val_gen, test_gen

class TrainedModel:
    def __init__(self, models_dir, model_type=ModelType.XCEPTION, input_shape=(299, 299, 3), fine_tune=False):
        self._model_type = model_type
        self._input_shape = input_shape
        self._fine_tune = fine_tune
        self._base_model = self._build_base_model()
        self._model = self._build_model()
        self._models_dir = os.path.join(models_dir, 'x')
        if self._model_type == ModelType.EFFICIENTNET:
            self._models_dir = os.path.join(models_dir, 'net')
        create_directory(self._models_dir)

    def _build_base_model(self):
        if self._model_type == ModelType.EFFICIENTNET:
            base_model = EfficientNetB0(
                weights='imagenet', 
                include_top=False,
                input_shape=self._input_shape
            )
        else:
            base_model = Xception(
                weights="imagenet",
                include_top=False,
                input_shape=self._input_shape
            )
        return base_model
    
    def _build_model(self):
        inputs = Input(shape=self._input_shape)
        
        x = self._base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer=Adam(),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ]
        )
        return model

    def predict(self, frames):
        """Extract features from the video frames"""
        if len(frames.shape) == 3:
            frames = np.expand_dims(frames, axis=0)

        if self._model_type == ModelType.EFFICIENTNET:
            processed_frames = tf.keras.applications.efficientnet.preprocess_input(frames)
        else:
            processed_frames = tf.keras.applications.xception.preprocess_input(frames)

        features = self._model.predict(processed_frames)
        return features

class FineTunedModel(TrainedModel):

    def __init__(self, models_dir, model_type=ModelType.XCEPTION, input_shape=(299, 299, 3)):
        super().__init__(models_dir, model_type, input_shape, fine_tune=True)

    def _build_model(self):
        inputs = Input(shape=self._input_shape)
        
        x = self._base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer=Adam(),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ]
        )
        return model
    
    def fit(self, train_generator, val_generator, initial_epochs=10, fine_tune_epochs=10):
        """
        Двухэтапное обучение с возможностью дообучения
    
        Parameters:
            train_generator: генератор тренировочных данных
            val_generator: генератор валидационных данных
            fine_tune: выполнять ли дообучение
            initial_epochs: количество эпох начального обучения
            fine_tune_epochs: количество эпох дообучения
        """
        
        history = self._model.fit(
            train_generator,
            epochs=initial_epochs,
            validation_data=val_generator,
            callbacks=[
                ModelCheckpoint(
                    os.path.join(self._models_dir, "initial_model.h5"),
                    monitor="val_auc",
                    save_best_only=True,
                    mode="max"
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        
        if not self._fine_tune:
            return self._model, history
        
        self._base_model.trainable = True
        
        # 70% слоев заморозили
        for layer in self._base_model.layers[:int(len(self._base_model.layers)*0.7)]:
            layer.trainable = False
        
        self._model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ]
        )
        
        fine_tune_history = self._model.fit(
            train_generator,
            epochs=initial_epochs + fine_tune_epochs,
            initial_epoch=history.epoch[-1] + 1,
            validation_data=val_generator,
            callbacks=[
                ModelCheckpoint(
                    os.path.join(self._models_dir, "xception_finetune_deepfake_model.h5"),
                    monitor="val_auc",
                    save_best_only=True,
                    mode="max"
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        full_history = {
            k: history.history[k] + fine_tune_history.history[k]
            for k in history.history
        }
        
        return self._model, full_history