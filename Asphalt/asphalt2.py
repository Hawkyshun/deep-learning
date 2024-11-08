import os
import cv2 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Verilerin bulunduğu dizinler
base_dir = '/Users/furkanerdogan/Projects/deep-learning/Asphalt/449'
cracks_dir = os.path.join(base_dir, 'Cracks')
non_cracks_dir = os.path.join(base_dir, 'NonCracks')

# Görüntü boyutu
IMG_SIZE = 224

# Verileri yükleme ve yeniden boyutlandırma
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    return images, labels

crack_images, crack_labels = load_images_from_folder(cracks_dir, 1)
non_crack_images, non_crack_labels = load_images_from_folder(non_cracks_dir, 0)

# Verileri birleştirme
images = np.array(crack_images + non_crack_images)
labels = np.array(crack_labels + non_crack_labels)

# Verileri eğitim ve doğrulama setlerine bölme
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Verileri normalize etme
X_train = X_train / 255.0
X_val = X_val / 255.0

# Etiketleri one-hot encoding ile dönüştürme
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

# CNN modelinin oluşturulması
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

# Modelin derlenmesi
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modelin eğitilmesi
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# Modeli kaydetme
model.save('crack_detection_model.h5')
