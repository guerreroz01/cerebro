---
type: knowledge
tags:
  - ai
banner: "![[Pasted image 20231108230233.png]]"
banner_y: 0.47667
---
# Descenso de gradiente
___
Es un algoritmo de optimización utilizado para minimizar una función objetivo, que generalmente representa el error o la pérdida en un modelo de aprendizaje automático.

La idea básica detrás del descenso de gradiente es iterar de manera incremental los parámetros del modelo (_como los pesos en una red neuronal_) para encontrar el conjunto de parámetros que minimice la [[función de pérdida]].

>[!abstract] ¿Cuál es su función?
>En el contexto de las redes neuronales, el descenso de gradiente se utiliza durante el proceso de entrenamiento para ajustar los pesos de las conexiones entre las neuronas.

## Proceso:

### Inicialización:
Los pesos de la red neuronal se inicializan con valores pequeños aleatorios o por algún método de inicialización específico.

### Cálculo del Gradiente:
Para cada parámetro _peso_, se calcula el gradiente de la [[función de pérdida]] con respecto a ese parámetro. 

>[!note] ¿Que es el gradiente?
>El gradiente es un vector que apunta en la dirección del mayor incremento de la función de pérdida y cuya magnitud indica la pendiente de la función en ese punto.

### Actualización de Parámetros:
Los pesos se actualizan en la dirección opuesta al gradiente para reducir la [[función de pérdida]]. La actualización se realiza como sigue:

$w_{nuevo} = w_{viejo} - \eta \cdot \nabla{L(w_{viejo})}$

Donde ($w_{viejo}$) es el valor actual del peso, ($\nabla{L(w_{viejo})}$) es el gradiente de la función de pérdida respecto al peso, y ($\eta$) es la tasa de aprendizaje, un hiperparámetro que determina el tamaño del paso en la actualización.

### Iteración:
Este proceso se repite para múltiples iteraciones o épocas, con el cálculo del gradiente y la actualización de pesos llevándose a cabo para cada muestra o lote de muestras (_en el caso de variantes como el [[descenso de gradiente estocástico]] o el [[descenso de gradiente por lotes]] respectivamente_).

## Criterio de parada:
El descenso de gradiente continúa hasta que se cumple un criterio de parada, que puede ser:
1. Un número máximo de iteraciones.
2. Una mejora mínima en la [[función de pérdida]].
3. La convergencia a un umbral de error.

## Principales variantes del descenso de gradiente:
- [[descenso de gradiente por lotes|Descenso de gradiente por lotes]] (Batch Gradiente Descent): Utiliza todo el conjunto de datos para calcular el gradiente y actualizar los pesos en cada iteración.

- [[descenso de gradiente estocástico|Descenso de gradiente estocástico]] (Stochastic Gradient Descent, SGD): Actualiza los pesos después de cada muestra de entrenamiento. Es más rápido pero puede ser más ruidoso que el descenso de gradiente por lotes.

- [[Descenso de Gradientes por lotes miniatura]] (Mini-batch Gradient Descent): Compromiso entre los dos anteriores, actualiza los pesos después de un pequeño lote de muestras, equilibrando la eficiencia y la estabilidad del algoritmo.

# Referencias
___
- [[Red Neuronal]]
- [[función de pérdida]]
