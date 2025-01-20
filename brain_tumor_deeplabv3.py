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

def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1):
    x = Conv2D(num_filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)(block_input)
    x = BatchNormalization()(x)
    return tf.nn.relu(x)

def ASPP(inputs):
    # Atrous Spatial Pyramid Pooling
    dims = inputs.shape
    
    # Image pooling
    pool = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(inputs)
    pool = Conv2D(256, 1, padding='same')(pool)
    pool = BatchNormalization()(pool)
    pool = Activation('relu')(pool)
    pool = UpSampling2D(size=(dims[-3] // pool.shape[1], dims[-2] // pool.shape[2]), 
                       interpolation='bilinear')(pool)
    
    # 1x1 conv
    conv_1x1 = convolution_block(inputs, kernel_size=1)
    
    # 3x3 conv rate=6
    conv_3x3_1 = convolution_block(inputs, kernel_size=3, dilation_rate=6)
    
    # 3x3 conv rate=12
    conv_3x3_2 = convolution_block(inputs, kernel_size=3, dilation_rate=12)
    
    # 3x3 conv rate=18
    conv_3x3_3 = convolution_block(inputs, kernel_size=3, dilation_rate=18)
    
    # Concatenate
    concat = Concatenate()([pool, conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3])
    
    # Final 1x1 conv
    conv_final = Conv2D(256, 1, padding='same')(concat)
    conv_final = BatchNormalization()(conv_final)
    conv_final = Activation('relu')(conv_final)
    
    return conv_final

def DeepLabV3Plus(input_shape=(256, 256, 3)):
    # Encoder
    inputs = Input(shape=input_shape)
    
    # Entry flow
    x = Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Middle flow with dilated convolutions
    previous_block = x
    
    # First block with skip connection
    x = Conv2D(128, 3, padding='same', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, 3, padding='same', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    skip1 = x
    
    # ASPP block
    x = ASPP(x)
    
    # Decoder
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    
    # Skip connection processing
    skip_features = Conv2D(48, 1, padding='same')(skip1)
    skip_features = BatchNormalization()(skip_features)
    skip_features = Activation('relu')(skip_features)
    
    # Concatenate skip features
    x = Concatenate()([x, skip_features])
    
    # Final convolutions
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Final upsampling
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    
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
    model = DeepLabV3Plus()
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
            tf.keras.callbacks.ModelCheckpoint('deeplabv3plus_brain.h5', 
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
    np.save('deeplabv3plus_history.npy', history.history) 