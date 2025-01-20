import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, DenseNet121

def segnet_model(input_shape=(256, 256, 3), n_classes=1):
    # Encoder
    inputs = Input(input_shape)
    
    # Encoder blocks
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Decoder blocks
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def deeplabv3plus_model(input_shape=(256, 256, 3), n_classes=1):
    inputs = Input(input_shape)
    
    # Backbone with atrous convolution
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Atrous Spatial Pyramid Pooling
    # Rate = 6
    b1 = Conv2D(256, (3, 3), padding='same', dilation_rate=6)(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    
    # Rate = 12
    b2 = Conv2D(256, (3, 3), padding='same', dilation_rate=12)(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    
    # Rate = 18
    b3 = Conv2D(256, (3, 3), padding='same', dilation_rate=18)(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation('relu')(b3)
    
    # Image pooling
    b4 = GlobalAveragePooling2D()(x)
    b4 = Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, 1), 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same')(b4)
    
    # Merge branches
    x = Concatenate()([b1, b2, b3, b4])
    x = Conv2D(256, (1, 1), padding='same')(x)
    
    # Upsampling
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def resnet_fcn(input_shape=(256, 256, 3), n_classes=1):
    # ResNet50 backbone
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Get feature maps from different stages
    c1 = base_model.get_layer('conv2_block3_out').output
    c2 = base_model.get_layer('conv3_block4_out').output
    c3 = base_model.get_layer('conv4_block6_out').output
    c4 = base_model.get_layer('conv5_block3_out').output
    
    # FCN head
    x = Conv2D(512, (3, 3), padding='same')(c4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Upsampling path
    x = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(x)
    
    return Model(inputs=base_model.input, outputs=outputs)

def densenet_aspp(input_shape=(256, 256, 3), n_classes=1):
    # DenseNet backbone
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Get the output of the last DenseNet block
    x = base_model.output
    
    # ASPP module
    b1 = Conv2D(256, (1, 1), padding='same')(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    
    b2 = Conv2D(256, (3, 3), padding='same', dilation_rate=6)(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    
    b3 = Conv2D(256, (3, 3), padding='same', dilation_rate=12)(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation('relu')(b3)
    
    b4 = Conv2D(256, (3, 3), padding='same', dilation_rate=18)(x)
    b4 = BatchNormalization()(b4)
    b4 = Activation('relu')(b4)
    
    # Merge ASPP branches
    x = Concatenate()([b1, b2, b3, b4])
    
    # Upsampling
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(x)
    
    return Model(inputs=base_model.input, outputs=outputs)

def pspnet_model(input_shape=(256, 256, 3), n_classes=1):
    inputs = Input(input_shape)
    
    # Convolutional blocks
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Pyramid Pooling Module
    pool_sizes = [1, 2, 3, 6]
    pool_features = []
    
    for pool_size in pool_sizes:
        pool_feature = AveragePooling2D(pool_size=(pool_size, pool_size))(x)
        pool_feature = Conv2D(128, (1, 1), padding='same')(pool_feature)
        pool_feature = BatchNormalization()(pool_feature)
        pool_feature = Activation('relu')(pool_feature)
        pool_feature = UpSampling2D(
            size=(pool_size, pool_size), 
            interpolation='bilinear'
        )(pool_feature)
        pool_features.append(pool_feature)
    
    # Concatenate pyramid features
    x = Concatenate()(pool_features)
    
    # Final convolution blocks
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Upsampling to original size
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def unet_model(input_shape=(256, 256, 3), n_classes=1):
    # Input
    inputs = Input(input_shape)
    
    # Encoder path (Contracting)
    # Block 1
    conv1 = Conv2D(64, 3, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, 3, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2
    conv2 = Conv2D(128, 3, padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, 3, padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 3
    conv3 = Conv2D(256, 3, padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, 3, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Block 4
    conv4 = Conv2D(512, 3, padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, 3, padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bridge
    conv5 = Conv2D(1024, 3, padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, 3, padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    # Decoder path (Expanding)
    # Block 6
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    concat6 = Concatenate()([up6, conv4])
    conv6 = Conv2D(512, 3, padding='same')(concat6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, 3, padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    
    # Block 7
    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    concat7 = Concatenate()([up7, conv3])
    conv7 = Conv2D(256, 3, padding='same')(concat7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, 3, padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    
    # Block 8
    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    concat8 = Concatenate()([up8, conv2])
    conv8 = Conv2D(128, 3, padding='same')(concat8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, 3, padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    
    # Block 9
    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    concat9 = Concatenate()([up9, conv1])
    conv9 = Conv2D(64, 3, padding='same')(concat9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, 3, padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    # Output
    outputs = Conv2D(n_classes, 1, activation='sigmoid')(conv9)
    
    return Model(inputs=inputs, outputs=outputs)

# Model training function
def train_model(model, train_data, val_data, epochs=50):
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Dice()]
    )
    
    return model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
        ]
    ) 