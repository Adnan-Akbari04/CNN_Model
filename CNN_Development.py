import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_data(path:str):
    labels = []
    X = []

    for item in glob.glob(path):
        label = item.split('\\')[1]
        labels.append(label)
        img = cv2.imread(item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64,64))
        img = img / 255.0
        X.append(img)

    lb = LabelBinarizer()
    label_encoded = lb.fit_transform(labels)

    X = np.array(X, dtype=np.float32)

    return X, label_encoded, lb


# Load data
X, labels, lb = load_data('AugmentedDataSet\\*\\*\\*')
# print(f"Total images loaded: {len(X)}")

# Split data (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, shuffle=True, random_state=42)
# print(y_test.shape)

num_class = y_test.shape[1]
# print(num_class)

#region Create sequential model
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))
# model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

# Third Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

# Flatten the feature maps
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_class, activation='softmax'))  # Output layer (11 classes)


# Display model architecture
model.summary()

# Compile with optimizer, loss function, and metrics
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully")

# early_stop = EarlyStopping(
#     monitor='val_loss',
#     patience=10,  # Stop if no improvement for 10 epochs
#     restore_best_weights=True,
#     verbose=1
# )

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,              # Number of passes over the data
    batch_size=32,          # Number of samples per gradient update
    # callbacks=[early_stop],
    verbose=1               # Show progress bars
)

# endregion


# region Plot

for key,value in history.history.items():
  print(key, value)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('Results\\ModelResults.png', dpi=300, bbox_inches='tight')
plt.show()


y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)


# ConfusionMatrix
class_names = lb.classes_
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig('Results\\ModelsConfusionMatrix', dpi=300)
plt.show()

report = classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=class_names,
    output_dict=True
)

# Precision, Recall, F1-Score PLOT
df = pd.DataFrame(report).transpose()

df[['precision', 'recall', 'f1-score']][:-3].plot(kind='bar', figsize=(12,6))

plt.title("Precision, Recall, F1-score per Class")
plt.ylabel("Score")
plt.xticks(rotation=45)

plt.savefig("Results\\ClassificationReport.png", dpi=300)
plt.show()

# MissClassified Plot
misclassified = np.where(y_true_classes != y_pred_classes)[0]

plt.figure(figsize=(15,10))

for i, idx in enumerate(misclassified[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[idx])

    plt.title(
        f"True: {class_names[y_true_classes[idx]]}\nPred: {class_names[y_pred_classes[idx]]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.savefig("Results\\MisclassifiedSamples.png", dpi=300)
plt.show()
# endregion

# Save Model
model.save('ExeProject\\CNN_Model.keras')
import pickle

with open('ExeProject\\class_names.pkl', 'wb') as f:
    pickle.dump(lb.classes_, f)

