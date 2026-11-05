# Estimación de edad mediante Visión Artificial con ResNet50

## Descripción del proyecto

Este proyecto explora la aplicación de técnicas de Deep Learning y visión artificial para estimar la edad de una persona a partir de imágenes faciales.

Se utiliza una red neuronal convolucional ResNet50, preentrenada con ImageNet, adaptada a una tarea de regresión mediante Transfer Learning.

El objetivo es desarrollar un modelo capaz de aprender características visuales relacionadas con la edad y evaluar su capacidad de generalización.

## Objetivos

- Implementar un modelo de visión artificial basado en redes neuronales convolucionales.
- Aplicar técnicas de Transfer Learning y Fine-Tuning.
- Preparar y procesar imágenes para el entrenamiento.
- Experimentar con técnicas de regularización y aumento de datos.
- Evaluar el desempeño predictivo y analizar sus limitaciones.

## Dataset

Se utiliza el conjunto de datos [UTKFace](https://susanqq.github.io/UTKFace/), que contiene más de 20,000 imágenes faciales con edades comprendidas entre 0 y 116 años.

El conjunto incluye anotaciones de edad, género y grupo demográfico. En este proyecto se utiliza exclusivamente la edad como variable objetivo.

Las etiquetas se extraen de los nombres de los archivos.

**Nota:** UTKFace se distribuye para investigación no comercial.

## Metodología

### 1. Preparación de datos

- Lectura de imágenes en formato JPG.
- Extracción de las etiquetas de edad.
- Construcción de un DataFrame con Pandas.
- Redimensionamiento de imágenes a 224 × 224 píxeles.
- Separación del conjunto de datos en entrenamiento (80 %) y validación (20 %).

### 2. Aumento de datos

Se utilizan transformaciones como rotaciones, desplazamientos, zoom y reflejos horizontales para aumentar la variabilidad de las imágenes durante el entrenamiento.

La configuración actual también aplica transformaciones al conjunto de validación; esto se identifica como un aspecto por corregir.

### 3. Arquitectura del modelo

La arquitectura implementada consta de:

- **ResNet50:** modelo convolucional preentrenado en ImageNet, sin su capa de clasificación original.
- **GlobalAveragePooling2D:** reducción de mapas de características.
- **Dense (1024 neuronas):** capa completamente conectada con activación ReLU y regularización L2.
- **Dense (1 neurona):** salida numérica para estimar la edad.

Se plantea utilizar Fine-Tuning sobre las últimas 50 capas de ResNet50. La implementación actual requiere un ajuste para congelar explícitamente las capas anteriores.

### 4. Entrenamiento

Configuración inicial:

| Parámetro | Valor |
|---|---|
| Optimizador | Adam |
| Learning rate | 0.0001 |
| Función de pérdida | Huber Loss |
| Batch size | 32 |
| Épocas máximas | 10 |
| Regularización | L2 (0.01) |

Se incluyen los callbacks EarlyStopping y ReduceLROnPlateau.

### 5. Evaluación

El modelo utiliza Huber Loss como función de entrenamiento y evaluación.

Como siguiente mejora, se incorporará el Error Absoluto Medio (MAE), que permite interpretar el error promedio de predicción en años, junto con un conjunto de prueba independiente.

**Resultados cuantitativos:** pendientes de verificar mediante una ejecución reproducible del modelo.

## Tecnologías utilizadas

- Python
- TensorFlow / Keras
- ResNet50
- Pandas
- NumPy

## Estructura del repositorio

- `README.md`: documentación del proyecto.
- `modelo_edad.py`: procesamiento, construcción, entrenamiento y evaluación del modelo.

## Ejecución

El proyecto requiere Python y las dependencias mencionadas, además de descargar previamente el dataset UTKFace.

La implementación actual utiliza una ruta local para acceder a las imágenes. Antes de ejecutarlo en otro equipo, es necesario actualizar esa ruta y configurar un entorno compatible de TensorFlow.

## Limitaciones y trabajo futuro

- Corregir el proceso de congelamiento y descongelamiento de capas.
- Aplicar el preprocesamiento específico de ResNet50.
- Separar las transformaciones de entrenamiento y validación.
- Incorporar métricas MAE y RMSE.
- Implementar una evaluación final sobre datos de prueba independientes.
- Analizar el desempeño por intervalos de edad y grupos demográficos.
- Evaluar posibles sesgos y errores derivados de las etiquetas del dataset.
- Guardar el modelo entrenado y desarrollar una interfaz para inferencia.

## Alcance

Proyecto educativo y experimental orientado al aprendizaje de Deep Learning y visión artificial.

Las edades estimadas deben interpretarse como predicciones aproximadas, no como determinaciones de la edad real de una persona.

## Referencias

- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [Keras — ResNet50](https://keras.io/api/applications/resnet/)
- [TensorFlow](https://www.tensorflow.org/)
