import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.applications import ResNet50

# Verilerin bulunduğu dizinler
train_dir = '/Users/furkanerdogan/Projects/deep-learning/Fruits/fruits/train'
test_dir = '/Users/furkanerdogan/Projects/deep-learning/Fruits/fruits/test'

# Görüntü boyutu
IMG_SIZE = 100
NUM_CLASSES = 10

# Veri artırma işlemleri
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,       
    width_shift_range=0.2,   
    height_shift_range=0.2,  
    horizontal_flip=True,    
    zoom_range=0.15          
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Transfer Learning (ResNet50'ye ince ayar yaparak)
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Katmanları dondurma
for layer in resnet_base.layers:
    layer.trainable = False

model = Sequential([
    resnet_base,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Modelin derlenmesi
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modelin eğitilmesi
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator
)

# Modeli .h5 formatında kaydetme
model.save("model_fruit_classifier_resnet.h5")

# Eğitim ve doğrulama kayıp ve doğruluk grafikleri
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend(loc='lower right')
plt.title('Model Doğruluğu')
plt.show()

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend(loc='upper right')
plt.title('Model Kaybı')
plt.show()

# Test verileri üzerinden tahminlerin yapılması
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Özellik çıkarımı (Feature Extraction)
feature_extractor = Model(inputs=model.input, outputs=model.get_layer("dense").output)
features = feature_extractor.predict(test_generator)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.show()

# Metriklerin hesaplanması
precision = precision_score(y_true, y_pred_classes, average='macro')
recall = recall_score(y_true, y_pred_classes, average='macro')
f1 = f1_score(y_true, y_pred_classes, average='macro')
accuracy = accuracy_score(y_true, y_pred_classes)

# Specificity hesaplama
TN = conf_matrix.sum() - (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix))
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
specificity = np.mean(TN / (TN + FP))

# Metriklerin yazdırılması
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Specificity: {specificity:.4f}')
