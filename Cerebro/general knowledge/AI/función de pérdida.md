---
type: knowledge
tags:
  - ai
---
# Función de pérdida
___
También conocida como función de costo, en inteligencia artificial y aprendizaje automático, es una métrica que mide qué tan bien un modelo de aprendizaje automático está realizando su tarea.

La función de pérdida toma como entrada las predicciones del modelo y los valores verdaderos (o etiquetas) y calcula un valor numérico que representa el error o la diferencia entre estas predicciones y los valores reales.

>[!dander] En pocas palabras
>Toma el valor real y el valor del modelo y luego calcula un valor numérico que representa el error

El objetivo del entrenamiento de un modelo es minimizar la función de pérdida; es decir, ajustar los parámetros del modelo (_como los pesos en una [[Red Neuronal|red neuronal]]_) para que las predicciones se acerquen lo máximo posible a los valores reales, reduciendo así el error.

## Tipos de funciones de pérdida:
Existen diferentes tipos de funciones de pérdida y la elección de una función específica dependerá del tipo de tarea que el modelo está intentando realizar (_por ejemplo, regresión, clasificación, segmentación, etc._) y de las características específicas del problema.

1. [[Error cuadrático medio]] (Mean Squared Error, MSE): utilizada comúnmente en tareas de regresión. Calcula el promedio de los cuadrados de las diferencias entre las predicciones y los valores verdaderos. $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ donde $(y_i)$ es el valor verdadero y $(\hat{y}_i)$ es la predicción del modelo para la i-ésima muestra.

2. [[Entropía cruzada]] (Cross-Entropy): Utilizada en tareas de clasificación. Mide la diferencia entre dos distribuciones de probabilidad: la distribución real (_verdaderas etiquetas_) y la distribución estimada por el modelo (_predicciones_). 
	1. Para clasificación binaria: $H(y, \hat{y}) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$. 
	2. Para clasificación multiclase: $H(y, \hat{y}) = -\sum_{i=1}^{n} \sum_{j=1}^{m} y_{ij} \log(\hat{y}_{ij})$ donde $(y_{ij})$ es si la muestra (i) pertenece a la clase (j) y $(\hat{y}_{ij})$ es la probabilidad predicha de que la muestra (i) pertenezca a la clase (j).
3. [[Función de pérdida de Hinge]] (Hinge Loss): Común en máquinas de vectores de soporte (SVM) para clasificación:
		1. Hinge Loss = $\frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i \cdot \hat{y}_i)$ donde donde $(y_i)$ es la etiqueta de clase (generalmente +1 o -1) y $(\hat{y}_i)$ es el valor predicho por el modelo.

Durante el entrenamiento se utiliza un algoritmo de optimización (_como el [[descenso de gradiente]]_) para minimizar la función de pérdida.

El valor de la función de pérdida actúa como una señal de retroalimentación para el algoritmo, indicando qué tan bien está funcionando el modelo y cómo  deberían ajustarse los parámetros para mejorar su rendimiento.

# Referencias
---
- [[descenso de gradiente]]
- [[Red Neuronal]]
