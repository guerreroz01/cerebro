---
type: knowledge
tags:
  - ai
  - redneuronal
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/uwbHOpp9xkc?si=M3bOlojuyVFcOUdP" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
# Red Neuronal
Las redes neuronales artificiales artificiales están compuestas de unidades básicas llamadas neuronas artificiales o nodos, las cuales están organizadas en capas y conectadas entre sí. Estas neuronas procesan señales de entrada y producen señales de salida. La información se transmite a través de la red mediante conexiones ponderadas (_que tienen peso o un valor_), donde cada conexión tiene un peso asociado que determina la importancia de la señal de la conexión.

![[Pasted image 20231108224524.png]]


## Partes de una red neuronal:
Una red neuronal típica incluye:
1. **Capa de entrada**: Recibe los datos de entrada para la red

2. **Capas ocultas**: Realizan el procesamiento y la transformación de los datos, a través de varias capas. Pueden ser pocas o muchas, dependiendo de la complejidad de la red.

3. **Capa de salida**: Proporciona la respuesta o salida de la red final.

![[Pasted image 20231113210202.png]]
Cuando se coloca una neurona de forma secuencial una de ellas recibe la información procesada por la neurona anterior. Con esto se consigue que la red neuronal pueda aprender a través de conocimiento jerarquizado.

Entre más capas añadimos más complejo puede ser el conocimiento que elaboremos, esta profundidad en la cantidad de capas es lo que da nombre al _**aprendizaje profundo**_ o [[Deep Learning]].

Las neuronas tienen que procesar el resultado con una [[Función de activación]]

## ¿Como se lleva a cabo el aprendizaje en las redes neuronales?
El aprendizaje se lleva a cabo ajustando los pesos de las conexiones en un proceso llamado entrenamiento. 

Durante el entrenamiento, se presenta a la red una serie de ejemplos con entradas y salidas conocidas. La red hace predicciones y luego compara sus predicciones con los resultados reales. A través de un proceso iterativo que involucra una función de pérdida y un algoritmo de optimización (como el [[descenso de gradiente]]), la red ajusta sus pesos para minimizar la diferencia entre las predicciones y los resultados reales.

___
# Referencias
- [[descenso de gradiente]]
- [[función de pérdida]]
- [[Función de activación]]
- [[backpropagation]]
- [[Deep Learning]]

