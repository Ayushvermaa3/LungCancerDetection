import os
import numpy as np
import tensorflow as tf
import pydicom
import cv2
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from datetime import datetime
from pydicom.uid import generate_uid, SecondaryCaptureImageStorage

def convert_jpg_to_dicom(image_path, dicom_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  

    if image is None:
        print(f"Skipping invalid image: {image_path}")
        return

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(dicom_path, {}, file_meta=file_meta, preamble=b"DICM" + b"\0" * 124)  

    ds.PatientName = "LungCancerPatient"
    ds.PatientID = "123456"
    ds.Modality = "CT"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.StudyDate = datetime.now().strftime('%Y%m%d')

    ds.Rows, ds.Columns = image.shape
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = image.tobytes()

    ds.save_as(dicom_path, write_like_original=False)


import pydicom
import os

dicom_path = "/Users/ayush/Desktop/The IQ-OTHNCCD DICOM/normal/"

for file in os.listdir(dicom_path):
    if file.endswith(".dcm"):
        try:
            dcm = pydicom.dcmread(os.path.join(dicom_path, file))
            print(f"{file} is a valid DICOM file")
        except Exception as e:
            print(f"{file} is corrupted: {e}")


data_path = "/Users/ayush/Desktop/The IQ-OTHNCCD lung cancer dataset/"
converted_path = "/Users/ayush/Desktop/The IQ-OTHNCCD DICOM/" 

os.makedirs(converted_path, exist_ok=True)

for category in ["normal", "benign", "malignant"]:
    input_folder = os.path.join(data_path, category)
    output_folder = os.path.join(converted_path, category)
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith(".jpg"):
            jpg_path = os.path.join(input_folder, file)
            dicom_filename = file.replace(".jpg", ".dcm")
            dicom_path = os.path.join(output_folder, dicom_filename)
            convert_jpg_to_dicom(jpg_path, dicom_path)

print("✅ JPG to DICOM conversion completed successfully!")

import os

data_path = "/Users/ayush/Desktop/The IQ-OTHNCCD DICOM/"


for category in ["normal", "benign", "malignant"]:
    folder_path = os.path.join(data_path, category)
    print(f"Checking {folder_path}...")

    for file in os.listdir(folder_path):
        if not file.lower().endswith(".dcm"):
            print(f"⚠️ Non-DICOM file detected: {file}")

import pydicom

def load_dicom_images(data_dir, img_size=(224, 224)):
    images, labels = [], []
    class_map = {"normal": 0, "benign": 1, "malignant": 2}

    for class_name in class_map.keys():
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            dicom_path = os.path.join(class_dir, file)

            if not file.lower().endswith(".dcm"):
                print(f"Skipping non-DICOM file: {file}")
                continue

            try:
                
                dicom_data = pydicom.dcmread(dicom_path, force=True)

                if "PixelData" not in dicom_data:
                    print(f"Skipping {dicom_path}: No PixelData found ❌")
                    continue

                image = dicom_data.pixel_array
                image = cv2.resize(image, img_size)  # Resize image
                image = image / 255.0  # Normalize pixel values

                images.append(image)
                labels.append(class_map[class_name])

            except Exception as e:
                print(f"Error reading {dicom_path}: {e}")
                continue  # Skip corrupted files

    return np.array(images), np.array(labels)


##print("🛠️ Loading DICOM images...")
X, y = load_dicom_images(data_path)


import os

data_path = "/Users/ayush/Desktop/The IQ-OTHNCCD DICOM/"

print(f"X shape: {X.shape}, y shape: {y.shape}")  # Debugging


data_path = "/Users/ayush/Desktop/The IQ-OTHNCCD DICOM/"

X, y = load_dicom_images(data_path)
X = np.expand_dims(X, axis=-1)  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from keras._tf_keras.keras.layers import Input 

def build_model(input_shape):
    model = Sequential([
    Input(shape=(224, 224, 1)), 
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  
])

    return model

model = build_model(X_train.shape[1:])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy during training')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during training')
plt.legend()
plt.show()


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

model.save("lung_cancer_model.keras") 

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import load_model

model_path = "/Users/ayush/Lung Cancer detection/lung_cancer_model.keras"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)

def load_jpg_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  
    img = np.expand_dims(img, axis=0)   
    return img


test_image_path = "/Users/ayush/Desktop/TEST CASE/Normal case (386).jpg"

if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"Test image not found at {test_image_path}")

test_image = load_jpg_image(test_image_path)

prediction = model.predict(test_image)
class_labels = ["Normal", "Benign", "Malignant"]
predicted_class = np.argmax(prediction, axis=1)[0]

print(f"🎯 Prediction: {class_labels[predicted_class]}")

plt.imshow(test_image[0, :, :, 0], cmap="gray")
plt.title(f"Prediction: {class_labels[predicted_class]}")
plt.axis("off")
plt.show()
