import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 📂 Ruta a las imágenes
images_path = r"C:\Users\utril\Desktop\Tripleten\vision artificial\caras"

# 🔹 Crear un DataFrame con nombres de archivos y edades
image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
# Extrae la edad del nombre del archivo
ages = [int(f.split('_')[0]) for f in image_files]

df = pd.DataFrame({'filename': image_files, 'age': ages})

# 🔹 Generador de imágenes con aumentación
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 🔹 Generador de datos de entrenamiento
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=images_path,  # Carpeta donde están las imágenes
    x_col='filename',
    y_col='age',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',  # Para regresión, no clasificación
    subset='training'
)

# 🔹 Generador de datos de validación
val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=images_path,
    x_col='filename',
    y_col='age',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    subset='validation'
)

# 🔹 Modelo Base: ResNet50 (sin capas superiores)
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))

# 🔹 Descongelar las últimas 50 capas para ajuste fino
for layer in base_model.layers[-50:]:
    layer.trainable = True

# 🔹 Añadir capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu',
          kernel_regularizer=regularizers.l2(0.01))(x)
output = Dense(1)(x)

# 🔹 Crear modelo final
model = Model(inputs=base_model.input, outputs=output)

# 🔹 Compilar el modelo con Huber Loss y Adam
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=tf.keras.losses.Huber())

# 🔹 Callbacks para mejorar el entrenamiento
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stop = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

# 🔹 Entrenar el modelo
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[reduce_lr, early_stop]
)

# 🔹 Evaluar el modelo en los datos de validación
mae = model.evaluate(val_generator, verbose=1)

# 🔹 Mostrar el MAE
print(f"🔹 Error Medio Absoluto (MAE) en el conjunto de validación: {mae}")
