import os
import zipfile
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="EiumqQQO33oWqjhBJiJo")
project = rf.workspace("nidha-rahiman").project("clothing-color-detection")
dataset = project.version(1).download("coco")

!pip install tensorflow numpy matplotlib

import os

# List files in the dataset folder
print(os.listdir("clothing-color-detection-1"))

import json

# Load the COCO annotations
with open("clothing-color-detection-1/train/_annotations.coco.json") as f:
    annotations = json.load(f)

# Print some annotation details
print(annotations.keys())  # Should contain 'images', 'annotations', 'categories'

# Create a dictionary mapping image IDs to file names
image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
print(image_id_to_filename)

# Create a dictionary mapping category IDs to color labels
category_id_to_label = {cat['id']: cat['name'] for cat in annotations['categories']}
print(category_id_to_label)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Load a sample image
sample_image_id = annotations['annotations'][425]['image_id']
sample_image_name = image_id_to_filename[sample_image_id]
image_path = f"clothing-color-detection-1/train/{sample_image_name}"

# Open and display the image
image = Image.open(image_path)
plt.figure(figsize=(6, 6))
plt.imshow(image)

# Add bounding boxes
ax = plt.gca()
for ann in annotations['annotations']:
    if ann['image_id'] == sample_image_id:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        label = category_id_to_label[ann['category_id']]
        plt.text(x, y, label, color='white', fontsize=12, backgroundcolor='red')

plt.show()

cropped_images = []
labels = []

for ann in annotations['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id']
    bbox = ann['bbox']  # Bounding box coordinates

    # Load the image
    img_path = f"clothing-color-detection-1/train/{image_id_to_filename[image_id]}"
    image = Image.open(img_path).convert("RGB")

    # Crop the image using bbox
    x, y, width, height = bbox
    cropped_image = image.crop((x, y, x + width, y + height))

    # Resize to 224x224 for training
    cropped_image = cropped_image.resize((224, 224))

    # Convert to numpy array
    image_array = np.array(cropped_image)

    # Store the processed image and label
    cropped_images.append(image_array)
    labels.append(category_id_to_label[category_id])

# Convert to NumPy arrays
cropped_images = np.array(cropped_images)
labels = np.array(labels)

print(f"Processed {len(cropped_images)} images for training.")

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Convert string labels to numeric values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)  # Converts 'red', 'blue' → [0, 1, 2, ...]

# Convert to one-hot encoding
num_classes = len(label_encoder.classes_)  # Number of unique colors
one_hot_labels = to_categorical(encoded_labels, num_classes)

print("Unique Classes:", label_encoder.classes_)  # Check mapping of labels
print(f"Encoded Labels Shape: {one_hot_labels.shape}")

from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(cropped_images, one_hot_labels, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(num_classes, activation='softmax')  # Output layer (multi-class)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Show model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,  # Increase for better results
    batch_size=32,
    verbose=1
)

# Evaluate the trained model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

model.save("color_model.keras")
print("Model saved successfully!")

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("/content/drive/My Drive/color_model.h5")

# Upload a new image
from google.colab import files
uploaded = files.upload()

# Load and preprocess the image
img_path = list(uploaded.keys())[0]
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the color
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)  # Get index of highest probability
predicted_color = label_encoder.inverse_transform([predicted_class])[0]  # Convert index to label

# Display the image with prediction
plt.imshow(img)
plt.axis("off")  # Hide axis
plt.title(f"Predicted Color: {predicted_color}", fontsize=14, color="blue")
plt.show()

print(f"Predicted Color: {predicted_color}")

from google.colab import drive
drive.mount('/content/drive')

model_save_path = '/content/drive/My Drive/color_model.keras'

model.save(model_save_path)

model.summary()
from tensorflow.keras.models import load_model

# Load the model
model = load_model(model_save_path)

# Recompile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model loaded and recompiled successfully!")
from tensorflow.keras.models import load_model

model_save_path = '/content/drive/My Drive/color_model.keras'

# Load the model
model = tf.keras.models.load_model(model_save_path)

# Recompile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model loaded and recompiled successfully!")
