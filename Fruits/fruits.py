import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Verilerin bulunduğu dizinler
train_dir = '/Users/furkanerdogan/Projects/deep-learning/Fruits/fruits/train'
test_dir = '/Users/furkanerdogan/Projects/deep-learning/Fruits/fruits/test'

# Görüntü boyutu
IMG_SIZE = 100
NUM_CLASSES = 10  # 10 sınıf olduğu varsayılıyor

# Klasör adlarına karşılık gelen etiketler
class_labels = {class_name: idx for idx, class_name in enumerate(os.listdir(train_dir))}

# Verileri yükleme ve yeniden boyutlandırma
def load_images_from_folder(folder):
    images = []
    labels = []
    for label_name in os.listdir(folder):  # Sınıflara göre alt klasörlerde dolaş
        class_folder = os.path.join(folder, label_name)
        label = class_labels[label_name]  # Klasör adını etiket olarak al
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # 100x100 olarak yeniden boyutlandırma
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Eğitim ve test verilerini yükleme
X_train, y_train = load_images_from_folder(train_dir)
X_test, y_test = load_images_from_folder(test_dir)

# Verileri normalize etme
X_train = X_train / 255.0
X_test = X_test / 255.0

# Etiketleri one-hot encoding ile dönüştürme
y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_categorical = to_categorical(y_test, num_classes=NUM_CLASSES)

# Veri artırma
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# CNN modelinin oluşturulması
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Modelin derlenmesi
model.compile(optimizer=RMSprop(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Modelin eğitilmesi
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=15, validation_data=(X_test, y_test_categorical))

# Modeli kaydetme
model.save('fruit_classification_model_optimized_2.h5')

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

# Tahminlerin yapılması
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:\n", conf_matrix)

# Confusion matrix'i sayısal değerlerle görselleştirme
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.show()

# Metriklerin hesaplanması
precision = precision_score(y_test, y_pred_classes, average='macro')
recall = recall_score(y_test, y_pred_classes, average='macro')
f1 = f1_score(y_test, y_pred_classes, average='macro')
accuracy = accuracy_score(y_test, y_pred_classes)

# Specificity hesaplama (Her sınıf için)
TN = conf_matrix.sum() - (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix))
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
specificity = np.mean(TN / (TN + FP))

# Metriklerin yazdırılması
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Specificity: {specificity:.4f}')
