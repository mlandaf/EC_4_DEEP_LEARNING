<div align="center">

# 🍎 FruitSeg30 - Deep Learning para Clasificación de Frutas

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" alt="Status">
</p>

<p align="center">
  <strong>Proyecto de Deep Learning para clasificación y búsqueda de similaridad de 30 tipos de frutas utilizando múltiples arquitecturas de redes neuronales.</strong>
</p>

[Descripción](#-descripción) •
[Características](#-características) •
[Arquitecturas](#-arquitecturas-implementadas) •
[Instalación](#-instalación) •
[Uso](#-uso) •
[Resultados](#-resultados) •
[Autores](#-autores)

</div>

---

## 📋 Descripción

Este proyecto implementa un sistema completo de **clasificación de imágenes de frutas** utilizando técnicas avanzadas de Deep Learning. Se exploran y comparan múltiples arquitecturas, desde CNNs tradicionales hasta Transformers visuales y Redes Siamesas, proporcionando un análisis exhaustivo del rendimiento de cada enfoque.

### 🎯 Objetivos del Proyecto

- Implementar y comparar diversas arquitecturas de Deep Learning para clasificación de imágenes
- Explorar técnicas de Transfer Learning con modelos preentrenados en ImageNet
- Desarrollar un sistema de búsqueda de similaridad basado en embeddings
- Evaluar el rendimiento de clasificadores tradicionales (XGBoost) sobre representaciones aprendidas

---

## ✨ Características

| Característica | Descripción |
|----------------|-------------|
| 🧠 **Multi-arquitectura** | CNN desde cero, Swin Transformer, Redes Siamesas |
| 🔄 **Transfer Learning** | Modelos preentrenados en ImageNet (ResNet18, Swin-T) |
| 📊 **Data Augmentation** | Transformaciones avanzadas para entrenamiento robusto |
| 🔍 **Búsqueda de Similaridad** | Sistema Top-K basado en embeddings y KNN |
| 📈 **Métricas Completas** | Accuracy, Precision, Recall, F1-Score, Matrices de Confusión |
| ⚡ **GPU Accelerated** | Soporte completo para CUDA |

---

## 📊 Dataset: FruitSeg30

El dataset **FruitSeg30** es un conjunto de datos público diseñado para tareas de clasificación y segmentación de frutas.

### Especificaciones

| Característica | Valor |
|---------------|-------|
| **Total de imágenes** | 1,969 |
| **Número de clases** | 30 |
| **Resolución original** | 512 × 512 píxeles |
| **Formato de imágenes** | JPG |
| **Formato de máscaras** | PNG |

### 🍇 Clases del Dataset

<details>
<summary>Ver todas las 30 categorías de frutas</summary>

| Categoría | Subcategorías |
|-----------|---------------|
| **🍎 Manzanas** | Apple_Gala, Apple_Golden Delicious |
| **🥭 Mangos** | Mango_Alphonso, Mango_Amrapali, Mango_Bari, Mango_Himsagar, Mango Golden Queen |
| **🍌 Frutas tropicales** | Avocado, Banana, Pineapple, Dragon, Carambola, Green Coconut, Guava |
| **🍇 Bayas y uvas** | Berry, Grape, Burmese Grape, Lichi |
| **🍊 Cítricos** | Orange, Malta, Kiwi |
| **🫒 Frutas de hueso** | Date Palm, Palm, Olive, Hog Plum, Persimmon |
| **🍉 Otras frutas** | Pomegranate, Watermelon, White Pear, Elephant Apple |

</details>

### 📁 Estructura del Dataset

```
data/
├── Apple_Gala/
│   ├── Images/          # Imágenes RGB en formato JPG
│   └── Mask/            # Máscaras de segmentación en PNG
├── Banana/
│   ├── Images/
│   └── Mask/
└── ... (30 carpetas de frutas)
```

---

## 🏗️ Arquitecturas Implementadas

### 1️⃣ CNN desde Cero

Arquitectura convolucional personalizada inspirada en VGG:

```
Input (224×224×3)
    ↓
[Conv Block 1] → 32 filtros → BatchNorm → ReLU → MaxPool → Dropout(0.1)
    ↓
[Conv Block 2] → 64 filtros → BatchNorm → ReLU → MaxPool → Dropout(0.15)
    ↓
[Conv Block 3] → 128 filtros → BatchNorm → ReLU → MaxPool → Dropout(0.2)
    ↓
[Conv Block 4] → 256 filtros → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
[Conv Block 5] → 256 filtros → BatchNorm → ReLU → MaxPool → Dropout(0.3)
    ↓
Global Average Pooling → FC → 30 clases
```

### 2️⃣ Swin Transformer (Transfer Learning)

- **Backbone:** Swin-T preentrenado en ImageNet-1K
- **Fine-tuning:** Todas las capas
- **Cabeza personalizada:** 768 → 512 → 30

### 3️⃣ Redes Siamesas

Arquitectura para aprendizaje de embeddings con múltiples configuraciones:

| Variante | Función de Pérdida | Clasificador |
|----------|-------------------|--------------|
| Siamesa v1 | Contrastive Loss | Fully Connected |
| Siamesa v2 | Contrastive Loss | XGBoost |
| Siamesa v3 | Triplet Loss | Fully Connected |
| Siamesa v4 | Triplet Loss | XGBoost |

**Características:**
- **Backbone:** ResNet18 preentrenado
- **Embedding dimension:** 128
- **Contrastive margin:** 2.0
- **Triplet margin:** 1.0

---

## ⚙️ Instalación

### Prerrequisitos

- Python 3.8 o superior
- CUDA 11.x (opcional, para aceleración GPU)
- Git

### Pasos de Instalación

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/tu-usuario/EC_4_DEEP_LEARNING.git
   cd EC_4_DEEP_LEARNING
   ```

2. **Crear entorno virtual** (recomendado)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Descargar el dataset**
   
   Descargar FruitSeg30 y extraer en la carpeta `data/`

---

## 🚀 Uso

### Ejecutar el Notebook

```bash
jupyter notebook trabajo.ipynb
```

### Estructura del Notebook

1. **Setup e Importaciones** - Configuración del entorno
2. **Carga de Datos** - Dataset y DataLoaders
3. **CNN desde Cero** - Entrenamiento y evaluación
4. **Swin Transformer** - Transfer Learning
5. **Redes Siamesas** - Contrastive y Triplet Loss
6. **Buscador de Similaridad** - Sistema Top-10 KNN
7. **Comparación de Resultados** - Análisis comparativo

---

## 📈 Protocolo Experimental

### División de Datos

| Conjunto | Porcentaje | Propósito |
|----------|------------|-----------|
| **Entrenamiento** | 70% | Aprendizaje de parámetros |
| **Validación** | 15% | Ajuste de hiperparámetros |
| **Test** | 15% | Evaluación final |

### Configuración de Entrenamiento

| Parámetro | CNN | Swin Transformer | Redes Siamesas |
|-----------|-----|------------------|----------------|
| **Épocas** | 50 | 30 | 50 |
| **Learning Rate** | 1e-3 | 1e-4 | 1e-4 |
| **Batch Size** | 32 | 32 | 32 |
| **Optimizer** | AdamW | AdamW | AdamW |
| **Scheduler** | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |

### Data Augmentation

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 📊 Resultados

### Métricas de Evaluación

| Métrica | Descripción |
|---------|-------------|
| **Accuracy** | Proporción de predicciones correctas |
| **Precision** | TP / (TP + FP) por clase |
| **Recall** | TP / (TP + FN) por clase |
| **F1-Score** | Media armónica de Precision y Recall |
| **Precision@K** | Precisión en búsqueda de similaridad (Top-10) |

### Pipeline Experimental

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE EXPERIMENTAL                        │
├─────────────────────────────────────────────────────────────────┤
│  Dataset FruitSeg30 (1969 imágenes, 30 clases)                 │
│           ↓                                                     │
│  Split: Train (70%) / Val (15%) / Test (15%)                   │
│           ↓                                                     │
│  ┌───────────────┬───────────────┬───────────────┐             │
│  │   CNN desde   │     Swin      │    Redes      │             │
│  │     cero      │  Transformer  │   Siamesas    │             │
│  └───────┬───────┴───────┬───────┴───────┬───────┘             │
│          ↓               ↓               ↓                      │
│  ┌───────────────────────────────────────────────┐             │
│  │           Clasificación (30 clases)           │             │
│  └───────────────────────────────────────────────┘             │
│          ↓               ↓               ↓                      │
│  ┌───────────────────────────────────────────────┐             │
│  │   Evaluación: Accuracy, F1, Confusion Matrix  │             │
│  └───────────────────────────────────────────────┘             │
│                          ↓                                      │
│  ┌───────────────────────────────────────────────┐             │
│  │      Buscador de Similaridad (Top-10 KNN)     │             │
│  └───────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tecnologías Utilizadas

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-337AB7?style=flat-square&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=flat-square&logo=seaborn&logoColor=white" alt="Seaborn">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

---

## 🔬 Reproducibilidad

Para garantizar la reproducibilidad de los experimentos:

- **Semilla fija:** 42 para todas las operaciones aleatorias
- **Determinismo CUDA:** `torch.backends.cudnn.deterministic = True`
- **Entorno:** PyTorch 2.x, CUDA (si disponible)

---

## 📁 Estructura del Proyecto

```
EC_4_DEEP_LEARNING/
├── 📓 trabajo.ipynb      # Notebook principal con todo el código
├── 📖 README.md          # Documentación del proyecto
├── 📋 requirements.txt   # Dependencias del proyecto
└── 📂 data/              # Dataset FruitSeg30
    ├── Apple_Gala/
    ├── Banana/
    └── ...
```

---

## 👥 Autores

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/mlandaf">
        <strong>Marcelo Landa</strong>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/GusGus27">
        <strong>Gustavo Uceda</strong>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/KGP2005">
        <strong>Kotler Garay</strong>
      </a>
    </td>
  </tr>
</table>

---