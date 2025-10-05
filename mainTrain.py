import cv2 # type: ignore
import os
import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from keras.utils import normalize # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense # type: ignore
from keras.utils import to_categorical # type: ignore

# Définir les chemins
image_directory = 'datasets/'
no_tumor_path = os.path.join(image_directory, 'no')
yes_tumor_path = os.path.join(image_directory, 'yes')

# Vérifier si les dossiers existent
if not os.path.exists(no_tumor_path):
    raise FileNotFoundError(f"Erreur : Le dossier spécifié est introuvable : {no_tumor_path}")
if not os.path.exists(yes_tumor_path):
    raise FileNotFoundError(f"Erreur : Le dossier spécifié est introuvable : {yes_tumor_path}")

# Lister les fichiers dans chaque dossier
no_tumor_images = os.listdir(no_tumor_path)
yes_tumor_images = os.listdir(yes_tumor_path)

print(f"Nombre d'images sans tumeur : {len(no_tumor_images)}")
print(f"Nombre d'images avec tumeur : {len(yes_tumor_images)}")

# Initialisation des datasets et labels
datasets = []
labels = []
INPUT_SIZE = 64

# Charger les images du dossier "no"
for image_name in no_tumor_images:
    if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):  # Prise en charge des formats courants
        image_path = os.path.join(no_tumor_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:  # Vérifier que l'image a été correctement lue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir en format RGB
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))  # Redimensionner
            datasets.append(image)
            labels.append(0)  # Label 0 pour "pas de tumeur"

# Charger les images du dossier "yes"
for image_name in yes_tumor_images:
    if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(yes_tumor_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
            datasets.append(image)
            labels.append(1)  # Label 1 pour "tumeur présente"

# Conversion en numpy arrays
datasets = np.array(datasets) / 255.0  # Normalisation des données entre 0 et 1
labels = np.array(labels)

# Division des données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2, random_state=42)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

# Conversion des labels en catégories
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test,  num_classes=2)


# Construire le modèle
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(
    x_train, y_train,
    batch_size=16,
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test),
    shuffle=True
)

# Sauvegarder le modèle
model.save('BrainTumor10Epoch.h5')
print("Modèle sauvegardé sous 'BrainTumor10Epoch.h5'.")

# Évaluer le modèle
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Précision sur l'ensemble de test : {accuracy * 100:.2f}%")


