import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob

# Veri yükleme ve ön işleme fonksiyonları
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

# ResNet50 tabanlı segmentasyon modeli
def build_resnet50_unet(input_shape=(256, 256, 3)):
    # Encoder - ResNet50
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Encoder çıktıları için katmanları al
    s1 = base_model.get_layer('conv1_relu').output      # 128x128
    s2 = base_model.get_layer('conv2_block3_out').output  # 64x64
    s3 = base_model.get_layer('conv3_block4_out').output  # 32x32
    s4 = base_model.get_layer('conv4_block6_out').output  # 16x16
    
    # Bridge
    b1 = base_model.get_layer('conv5_block3_out').output  # 8x8
    
    # Decoder
    d1 = UpSampling2D((2, 2))(b1)
    d1 = concatenate([d1, s4])
    d1 = Conv2D(512, (3, 3), padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    d1 = Dropout(0.3)(d1)
    
    d2 = UpSampling2D((2, 2))(d1)
    d2 = concatenate([d2, s3])
    d2 = Conv2D(256, (3, 3), padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    d2 = Dropout(0.3)(d2)
    
    d3 = UpSampling2D((2, 2))(d2)
    d3 = concatenate([d3, s2])
    d3 = Conv2D(128, (3, 3), padding='same')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Activation('relu')(d3)
    d3 = Dropout(0.2)(d3)
    
    d4 = UpSampling2D((2, 2))(d3)
    d4 = concatenate([d4, s1])
    d4 = Conv2D(64, (3, 3), padding='same')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Activation('relu')(d4)
    d4 = Dropout(0.2)(d4)
    
    # Final upsampling
    d5 = UpSampling2D((2, 2))(d4)
    d5 = Conv2D(32, (3, 3), padding='same')(d5)
    d5 = BatchNormalization()(d5)
    d5 = Activation('relu')(d5)
    
    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d5)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# Data generator
def data_generator(images, masks, batch_size=8):
    while True:
        ix = np.random.choice(len(images), batch_size)
        img_batch = np.array([preprocess_image(images[i]) for i in ix])
        mask_batch = np.array([preprocess_mask(masks[i]) for i in ix])
        yield img_batch, mask_batch

if __name__ == "__main__":
    # Model oluşturma
    model = build_resnet50_unet()
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
            tf.keras.callbacks.ModelCheckpoint('resnet50_unet_brain.h5', 
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
    np.save('resnet50_history.npy', history.history) 