# Customer Segmentation & Anomaly Detection using Self-Organizing Maps (SOM)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Unsupervised%20Learning-green.svg)](https://en.wikipedia.org/wiki/Unsupervised_learning)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Descripción del Proyecto
Este proyecto implementa una solución de **Machine Learning No Supervisado** para segmentar una base de datos de 2,240 clientes. Utilizando **Mapas Auto-organizados (SOM)**, logramos reducir la dimensionalidad de datos complejos para identificar patrones de comportamiento, perfiles socioeconómicos y detectar anomalías en los hábitos de consumo.

## 🧠 Metodología y Algoritmos

El flujo de trabajo se divide en las siguientes etapas clave:

1.  **Preprocesamiento de Datos:**
    * Imputación de valores faltantes en `Income` mediante la mediana.
    * Ingeniería de variables (Tratamiento de fechas y categorías).
    * Escalado robusto utilizando `StandardScaler` para asegurar la convergencia del SOM.
2.  **Modelado con SOM (Self-Organizing Maps):**
    * Entrenamiento de una red neuronal de 10x10 neuronas mediante la librería `MiniSom`.
    * Visualización de la **U-Matrix** para identificar la topología y densidad de los clústeres.
3.  **Clustering Secundario:**
    * Aplicación de `MiniBatchKMeans` sobre los pesos de la red SOM para definir 6 segmentos óptimos.
4.  **Detección de Anomalías:**
    * Uso del **Error de Cuantificación** para identificar el 1% de clientes con comportamientos más atípicos (Percentil 99).

## 📊 Segmentos Identificados

* **Clúster 2 & 4 (High Value):** Clientes con ingresos elevados y alta tasa de conversión.
* **Clúster 1 & 3 (Price Sensitive):** Familias jóvenes o con alta carga de dependientes, sensibles a promociones.
* **Clúster 5 (Nicho Oro):** Clientes de edad avanzada con un patrón de compra inusual en productos de lujo/oro.

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.x
* **Librerías principales:**
    * `MiniSom`: Implementación de la red neuronal SOM.
    * `Scikit-learn`: Preprocesamiento, escalado y clustering K-Means.
    * `Pandas` & `NumPy`: Manipulación de estructuras de datos.
    * `Matplotlib` & `Seaborn`: Visualización de U-Matrix y perfiles de clúster.

## 🚀 Cómo ejecutar el código

1. Clona el repositorio:
   ```bash
   git clone [https://github.com/jagoher/Customer-Segmentation-SOM.git](https://github.com/jagoher/Customer-Segmentation-SOM.git)
