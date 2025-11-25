# EC_4_DEEP_LEARNING
## Metodología Propuesta

### 1. Descripción del Dataset

#### 1.1 FruitSeg30 Dataset
El dataset **FruitSeg30** es un conjunto de datos público diseñado para tareas de clasificación y segmentación de frutas, compuesto por:

| Característica | Valor |
|---------------|-------|
| **Total de imágenes** | 1,969 |
| **Número de clases** | 30 |
| **Resolución original** | 512 × 512 píxeles |
| **Formato de imágenes** | JPG |
| **Formato de máscaras** | PNG |

#### 1.2 Clases del Dataset
Las 30 categorías de frutas incluidas son:

| Categoría | Subcategorías |
|-----------|---------------|
| **Manzanas** | Apple_Gala, Apple_Golden Delicious |
| **Mangos** | Mango_Alphonso, Mango_Amrapali, Mango_Bari, Mango_Himsagar, Mango Golden Queen |
| **Frutas tropicales** | Avocado, Banana, Pineapple, Dragon, Carambola, Green Coconut, Guava |
| **Bayas y uvas** | Berry, Grape, Burmese Grape, Lichi |
| **Cítricos** | Orange, Malta, Kiwi |
| **Frutas de hueso** | Date Palm, Palm, Olive, Hog Plum, Persimmon |
| **Otras frutas** | Pomegranate, Watermelon, White Pear, Elephant Apple |

#### 1.3 Estructura del Dataset
```
data/
├── [Nombre_Fruta]/
│   ├── Images/          # Imágenes RGB en formato JPG
│   └── Mask/            # Máscaras de segmentación en PNG
```

---

### 2. Protocolo Experimental

#### 2.1 División de Datos
Se implementa una división estratificada para mantener la proporción de clases en cada conjunto:

| Conjunto | Porcentaje | Propósito |
|----------|------------|-----------|
| **Entrenamiento** | 70% | Aprendizaje de parámetros del modelo |
| **Validación** | 15% | Ajuste de hiperparámetros y early stopping |
| **Test** | 15% | Evaluación final del rendimiento |

> **Nota:** Se utiliza `stratify` en `train_test_split` para garantizar representación proporcional de todas las clases.

#### 2.2 Preprocesamiento de Datos

**Transformaciones para Entrenamiento (Data Augmentation):**
- Redimensionamiento a 224 × 224 píxeles
- Flip horizontal aleatorio (p=0.5)
- Flip vertical aleatorio (p=0.3)
- Rotación aleatoria (±15°)
- Color Jitter (brillo, contraste, saturación, tono)
- Transformación afín aleatoria
- Normalización ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Transformaciones para Validación/Test:**
- Redimensionamiento a 224 × 224 píxeles
- Normalización ImageNet

#### 2.3 Configuración de Entrenamiento

| Parámetro | CNN | Swin Transformer | Redes Siamesas |
|-----------|-----|------------------|----------------|
| **Épocas** | 50 | 30 | 50 |
| **Learning Rate** | 1e-3 | 1e-4 | 1e-4 |
| **Batch Size** | 32 | 32 | 32 |
| **Optimizer** | AdamW | AdamW | AdamW |
| **Weight Decay** | 1e-3 | 1e-4 | 1e-4 |
| **Scheduler** | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |

#### 2.4 Arquitecturas Implementadas

##### A. CNN desde Cero
- Arquitectura simplificada tipo VGG con 5 bloques convolucionales
- Canales: 32 → 64 → 128 → 256 → 256
- Dropout2d progresivo (0.1 → 0.3) para regularización
- Global Average Pooling + Clasificador FC

##### B. Swin Transformer (Transfer Learning)
- Backbone: Swin-T preentrenado en ImageNet
- Fine-tuning completo de todas las capas
- Cabeza de clasificación personalizada (768 → 512 → 30)

##### C. Redes Siamesas
- **Backbone:** ResNet18 preentrenado
- **Dimensión de embedding:** 128
- **Funciones de pérdida:**
  - Contrastive Loss (margin=2.0)
  - Triplet Loss (margin=1.0)
- **Clasificadores downstream:**
  - Fully Connected (embedding → 256 → 128 → 30)
  - XGBoost (n_estimators=200, max_depth=6)

#### 2.5 Métricas de Evaluación

| Métrica | Descripción |
|---------|-------------|
| **Accuracy** | Proporción de predicciones correctas |
| **Precision** | TP / (TP + FP) por clase |
| **Recall** | TP / (TP + FN) por clase |
| **F1-Score** | Media armónica de Precision y Recall |
| **Matriz de Confusión** | Visualización de errores por clase |
| **Precision@K** | Precisión en búsqueda de similaridad (Top-10) |

#### 2.6 Reproducibilidad
- **Semilla fija:** 42 para todas las operaciones aleatorias
- **Determinismo CUDA:** `torch.backends.cudnn.deterministic = True`
- **Entorno:** PyTorch 2.x, CUDA (si disponible)

---

### 3. Resumen de Experimentos

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