import numpy as np
import tensorflow as tf
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

def encoder_block(inputs, n_filters, kernel_size=3):
    x = Conv2D(n_filters, kernel_size, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(n_filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    pool = MaxPooling2D(pool_size=(2, 2))(x)
    
    return pool, x

def decoder_block(inputs, skip_features, n_filters, kernel_size=3):
    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_features])
    
    x = Conv2D(n_filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(n_filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def SegNet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    
    # Encoder
    p1, c1 = encoder_block(inputs, 64)
    p2, c2 = encoder_block(p1, 128)
    p3, c3 = encoder_block(p2, 256)
    p4, c4 = encoder_block(p3, 512)
    
    # Bridge
    b1 = Conv2D(1024, (3, 3), padding='same')(p4)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    
    b2 = Conv2D(1024, (3, 3), padding='same')(b1)
    b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    
    # Decoder
    d1 = decoder_block(b2, c4, 512)
    d2 = decoder_block(d1, c3, 256)
    d3 = decoder_block(d2, c2, 128)
    d4 = decoder_block(d3, c1, 64)
    
    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def data_generator(images, masks, batch_size=8):
    while True:
        ix = np.random.choice(len(images), batch_size)
        img_batch = np.array([preprocess_image(images[i]) for i in ix])
        mask_batch = np.array([preprocess_mask(masks[i]) for i in ix])
        yield img_batch, mask_batch

if __name__ == "__main__":
    # Model oluşturma
    model = SegNet()
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
            tf.keras.callbacks.ModelCheckpoint('segnet_brain.h5', 
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
    np.save('segnet_history.npy', history.history) 