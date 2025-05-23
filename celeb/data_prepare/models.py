import datetime
import pandas as pd

import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
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
from tensorflow.keras.callbacks import TensorBoard
from enum import Enum
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_prepare.dataset_tools import create_directory

class ModelType(Enum):
    XCEPTION = "xception"
    EFFICIENTNET = "efficientnet"
    CUSTOM = "custom_cnn"

class TrainedModel:
    def __init__(self, model_type, input_shape):
        self._model_type = model_type
        self._input_shape = input_shape
        self._base_model = self._build_base_model()

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

class TunedModel(TrainedModel):
    def __init__(self, models_dir, results_dir, model_type, input_shape):
        super().__init__(model_type, input_shape)
        self._model = self._build_model()
        self._models_dir = models_dir
        self._results_dir = results_dir
        create_directory(models_dir)
        create_directory(results_dir)

    def _build_model(self):
        print("Pure model run")
        inputs = Input(shape=self._input_shape)
        
        x = self._base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(1, activation="sigmoid")(x)

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
    

    def evaluate(self, X_test):
        return self._model.evaluate(X_test)

    def load_trained_model(self):
        self._model = tf.keras.models.load_model(os.path.join(self._models_dir, self._funetune_model_name))

    def predict(self, X_test):
        return self._model.predict(X_test)


    
class PersistModel(TunedModel):
    def __init__(self, model, history_file_name='history.csv', ypred_file_name='y_pred.csv'):
        self._history_file_name = history_file_name
        self._ypred_file_name = ypred_file_name
        self._model = model
        self._results_dir = model._results_dir

    def fit(self, train_generator, val_generator, initial_epochs=10):
        print('Persist fitting run')
        model, history = self._model.fit(train_generator, val_generator, initial_epochs)
        pure_history_df = pd.DataFrame(history.history)
        pure_history_df.to_csv(f'{self._results_dir}/{self._history_file_name}', index=False)
        
        return model, history

    def predict(self, X_test):
        y_pred = self._model.predict(X_test)
        y_pred_df = pd.DataFrame(y_pred)
        y_pred_df.to_csv(f'{self._results_dir}/{self._ypred_file_name}', index=False)
        return y_pred

class PureModel(TunedModel):
    
    def __init__(self, models_dir, results_dir, funetune_model_name, model_type, input_shape):
        super().__init__(models_dir, results_dir, model_type, input_shape)
        self._funetune_model_name = funetune_model_name
        self._models_dir = models_dir


    def fit(self, train_generator, val_generator, initial_epochs=10):
        """
        Двухэтапное обучение с возможностью дообучения
    
        Parameters:
            train_generator: генератор тренировочных данных
            val_generator: генератор валидационных данных
            initial_epochs: количество эпох начального обучения
            fine_tune_epochs: количество эпох дообучения
        """

        log_dir = f"{self._models_dir}/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        print('Pure fitting run')
        history = self._model.fit(
            train_generator,
            epochs=initial_epochs,
            validation_data=val_generator,
            callbacks=[
                tensorboard_callback,
                ModelCheckpoint(
                    os.path.join(self._models_dir, self._funetune_model_name),
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
        
        return self._model, history

class XPureModel(PureModel):
    def __init__(self, models_dir, results_dir):
        print("XPure model run")
        results_dir = os.path.join(results_dir, 'xpure')
        x_models_dir = os.path.join(models_dir, 'xpure')
        model_type=ModelType.XCEPTION
        funetune_model_name = 'xpure_model.h5'
        input_shape = (299,299,3)
        super().__init__(x_models_dir, results_dir, funetune_model_name, model_type, input_shape)
        
class NetPureModel(PureModel):
    def __init__(self, models_dir, results_dir):
        print("NetPure model run")
        results_dir = os.path.join(results_dir, 'netpure')
        x_models_dir = os.path.join(models_dir, 'netpure')
        model_type=ModelType.EFFICIENTNET
        funetune_model_name = 'netpure_model.h5'
        input_shape = (224,224,3)
        super().__init__(x_models_dir, results_dir, funetune_model_name, model_type, input_shape)
        

class FineTunedModel(TunedModel):

    def __init__(self, models_dir, results_dir, funetune_model_name, model_type, input_shape):
        super().__init__(models_dir, results_dir, model_type, input_shape)
        self._funetune_model_name = funetune_model_name
        self._models_dir = models_dir
        create_directory(models_dir)

    def fit(self, train_generator, val_generator, initial_epochs=10, fine_tune_epochs=10):
        """
        Двухэтапное обучение с возможностью дообучения
    
        Parameters:
            train_generator: генератор тренировочных данных
            val_generator: генератор валидационных данных
            initial_epochs: количество эпох начального обучения
            fine_tune_epochs: количество эпох дообучения
        """

        log_dir = f"{self._models_dir}/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        history = self._model.fit(
            train_generator,
            epochs=initial_epochs,
            validation_data=val_generator,
            callbacks=[
                tensorboard_callback,
                ModelCheckpoint(
                    os.path.join(self._models_dir, self._funetune_model_name),
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
                tensorboard_callback,
                ModelCheckpoint(
                    os.path.join(self._models_dir, self._funetune_model_name),
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


class XFineTunedModel(FineTunedModel):
    def __init__(self, models_dir, results_dir):
        x_models_dir = os.path.join(models_dir, 'x')
        results_dir = os.path.join(results_dir, 'x')
        model_type=ModelType.XCEPTION
        input_shape=(299, 299, 3)
        funetune_model_name="xception_finetune_deepfake_model.h5"
        super().__init__(x_models_dir, results_dir, funetune_model_name, model_type, input_shape)


class NetFineTunedModel(FineTunedModel):
    def __init__(self, models_dir, results_dir):
        model_type=ModelType.EFFICIENTNET
        funetune_model_name="net_finetune_deepfake_model.h5"
        net_models_dir = os.path.join(models_dir, 'net')
        results_dir = os.path.join(results_dir, 'net')
        input_shape=(224, 224, 3)
        super().__init__(net_models_dir, results_dir, funetune_model_name, model_type, input_shape)


class FeatureExtractorModel(TrainedModel):
    def __init__(self, model_type=ModelType.XCEPTION, input_shape=(299, 299, 3)):
        super().__init__(model_type, input_shape)
        self._model = self._build_model()

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
        # if len(frames.shape) == 3:
        #     frames = np.expand_dims(frames, axis=0)
        # if self._model_type == ModelType.EFFICIENTNET:
        #     processed_frames = tf.keras.applications.efficientnet.preprocess_input(frames)
        # else:
        #     processed_frames = tf.keras.applications.xception.preprocess_input(frames)

        features = self._model.predict(frames)
        return features

class XFeatureExtractorModel(FeatureExtractorModel):
    def __init__(self):
        super().__init__(ModelType.XCEPTION, (299, 299, 3))
        

class CustomModel:
    def __init__(self, ex_feature_model, models_dir, model_name, learning_rate, input_dim=256, batch_size=32, epoch=10):
        self._models_dir = models_dir
        self._model_name = model_name
        self._batch_size = batch_size
        self._epoch = epoch
        self._learning_rate = learning_rate
        self._model = self._build_classifier(input_dim)

    def _build_classifier(self, input_dim):
        """Build binary classifier model"""
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self._learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        return model

    def fit(self, X_gen, v_gen):
        """Train the classifier"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                f"{self._models_dir}/{self._model_name}",
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.TensorBoard(log_dir='logs')
        ]

        history = self._model.fit(
            X_gen, 
            validation_data=v_gen,
            epochs=self._epoch,
            batch_size=self._batch_size,
            callbacks=callbacks
        )

        return history

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

class XBaseAndCustomModel(CustomModel):
    def __init__(self, models_dir, input_dim=256, batch_size=32, epoch=10):
        ex_feature_model = XFeatureExtractorModel()
        model_name='xbase_custom_model.h5'
        learning_rate=1e-4
        super().__init__(ex_feature_model, models_dir, model_name, learning_rate)

class FrameProcessor:
    def __init__(self, ex_feature_model, custom_model, data_loader):
        self._ex_feature_model = ex_feature_model
        self._custom_model = custom_model
        self._data_loader = data_loader

    def process(self):
        train, val, test = data_loader.load_genrators()
        train_frame_features = np.mean(self._ex_feature_model.predict(train), axis=0)
        val_frame_features = np.mean(self._ex_feature_model.predict(val), axis=0)
        test_frame_features = np.mean(self._ex_feature_model.predict(test), axis=0)
        
        history = self._custom_model.fit(train_frame_features, val_frame_features)
        loss, acc, auc = self._custom_model.evaluate(test_frame_features, y_test)

        return history, loss, acc, auc 
        