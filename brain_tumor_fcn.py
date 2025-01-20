import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob

def load_data(path):
    images = sorted(glob(os.path.join(path, 'images/*')))
    masks = sorted(glob(os.path.join(path, 'masks/*')))
    return images, masks

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    return img

def preprocess_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask

def FCN8(input_shape=(256, 256, 3)):
    # VGG16 backbone
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Skip connections
    pool3 = base_model.get_layer('block3_pool').output
    pool4 = base_model.get_layer('block4_pool').output
    pool5 = base_model.get_layer('block5_pool').output
    
    # FCN-8 architecture
    conv6 = Conv2D(4096, (7, 7), activation='relu', padding='same')(pool5)
    conv6 = Dropout(0.5)(conv6)
    conv7 = Conv2D(4096, (1, 1), activation='relu', padding='same')(conv6)
    conv7 = Dropout(0.5)(conv7)
    
    # Score layer
    score_final = Conv2D(1, (1, 1))(conv7)
    
    # 2x upsampling
    score2x = Conv2DTranspose(1, 4, strides=2, padding='same')(score_final)
    
    # Skip connection from pool4
    score_pool4 = Conv2D(1, (1, 1))(pool4)
    score_fused = Add()([score2x, score_pool4])
    
    # 2x upsampling
    score4x = Conv2DTranspose(1, 4, strides=2, padding='same')(score_fused)
    
    # Skip connection from pool3
    score_pool3 = Conv2D(1, (1, 1))(pool3)
    score_final = Add()([score4x, score_pool3])
    
    # 8x upsampling to original size
    upsample = Conv2DTranspose(1, 16, strides=8, padding='same', activation='sigmoid')(score_final)
    
    model = Model(inputs=base_model.input, outputs=upsample)
    return model

def data_generator(images, masks, batch_size=8):
    while True:
        ix = np.random.choice(len(images), batch_size)
        img_batch = np.array([preprocess_image(images[i]) for i in ix])
        mask_batch = np.array([preprocess_mask(masks[i]) for i in ix])
        yield img_batch, mask_batch

if __name__ == "__main__":
    # Model oluşturma
    model = FCN8()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
    )
    
    # Veri yükleme
    train_images, train_masks = load_data('data/train')
    val_images, val_masks = load_data('data/val')
    
    # Model eğitimi
    train_gen = data_generator(train_images, train_masks)
    val_gen = data_generator(val_images, val_masks)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_images) // 8,
        validation_data=val_gen,
        validation_steps=len(val_images) // 8,
        epochs=50,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('fcn8_brain.h5', 
                                             save_best_only=True,
                                             monitor='val_mean_io_u'),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5,
                                               monitor='val_mean_io_u',
                                               mode='max'),
            tf.keras.callbacks.EarlyStopping(patience=10,
                                           monitor='val_mean_io_u',
                                           mode='max')
        ]
    )
    
    # Eğitim geçmişini kaydet
    np.save('fcn8_history.npy', history.history) 