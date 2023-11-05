---
type: knowledge
tags:
  - algebralineal
  - ai
  - ai/algebrallineal
---
# Álgebra Lineal

## Motivación
Camino para aprender álgebra lineal con la intención de aplicar estos conocimientos para aprender inteligencia artificial

Conceptos necesarios:
- [x] Fundamentos de la álgebra lineal
- [ ] Vectores
- [ ] Matrices
- [ ] Espacios vectoriales
- [ ] Transformaciones lineales
- [ ] Transformaciones determinantes
- [ ] Autovalores
- [ ] Autovectores
- [ ] Determinantes
- [ ] Matrices inversas
- [ ] Descomposición de matrices

## Fundamentos:
El álgebra lineal se puede utilizar para calcular el movimiento de los objetos en el espacio y así poder predecir su trayectoria.

La álgebra lineal es una rama de las matemáticas que estudia conceptos tales como vectores, matrices, espacio dual, sistemas de ecuaciones lineales y su enfoque de manera más formal, espacios vectoriales y sus transformaciones lineales.

[[método de reducción gaussiana]] 

[[Clasificación de los números]]


## 1. Sistemas de ecuaciones lineales:
### 1.1 Sistemas con dos incógnitas:
Tienen la particularidad de que los puntos pueden visualizarse como los puntos de una recta. Es decir cada ecuación del sistema es la ecuación de una recta 

![[Pasted image 20231104231207.png]]

Para resolver este sistema por el método gráfico se dibujan en un mismo sistema de coordenadas , las tres rectas asociadas con las tres ecuaciones 

![[Pasted image 20231104231417.png]]

Un punto (a, b) es un punto solución del sistema solo si satisface cada ecuación, o lo que es lo mismo, solo si es un punto de cada una de las rectas (cada recta pasa por (a, b)). En nuestro ejemplo este punto es (1, 2).

Un sistema de ecuaciones lineales se interpreta también como una formulación matemática de un conjunto de restricciones, una por cada ecuación. Esto se comprende mejor si las variables tienen un significado físico como se ilustra en el ejemplo siguiente.

#### Ejemplo: 
En un acuarium hay dos especies de peces **A y B** que se alimentan con dos clases de alimentos. En la tabla siguiente se indican el consumo promedio diario de alimento de ambas especies y la cantidad de alimento disponible por clase (columna cuatro), en gramos.

![[Pasted image 20231104232120.png]]

El problema a resolver es calcular la cantidad máxima de peces de cada especie que pueden vivir en el acuarium.

2xA + xB = cantidad de alimento de la clase 1, consumida diariamente por la totalidad de los peces. Es decir : 2xA + xB = 25.

4xA + 3xB = cantidad de alimento de la clase 2, consumida diariamente por la totalidad de los peces. Es decir : 4xA + 3xB = 55.

La solución al problema planteado se encuentra resolviendo el siguiente sistema de ecuaciones lineales 2 × 2 :

![[Pasted image 20231104232535.png]]

De la primera ecuación se obtiene xB = 25−2xA. Esta expresión se sustituye en la segunda ecuación: 4xA + 3(25 − 2xA) = 55 y despejando xA, se tiene: xA = 10. Luego

xB = 25 − 2xA = 25 − 2(10) = 5.


