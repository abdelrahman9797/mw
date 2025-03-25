import numpy as np
import cv2
import os
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ضبط مسار البيانات
data_path = "C:/Users/Habib/Downloads/b-471/myData"
class_count = len(os.listdir(data_path))

images = []
labels = []

# استيراد الصور والفئات
print("Importing Images...")
for class_id in range(class_count):
    img_folder = os.path.join(data_path, str(class_id))
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))
        images.append(img)
        labels.append(class_id)
    print(f"Imported Class {class_id}")

# تحويل القوائم إلى مصفوفات NumPy
images = np.array(images)
labels = np.array(labels)

# معالجة البيانات
images = images / 255.0  # تحويل القيم إلى المجال [0, 1]
labels = to_categorical(labels, class_count)

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# بناء النموذج العصبي
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(class_count, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# حفظ النموذج المدرب
with open("model_trained.p", "wb") as file:
    pickle.dump(model, file)

print("Model training and saving completed!")
