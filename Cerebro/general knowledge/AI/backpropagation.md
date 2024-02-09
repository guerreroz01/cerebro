---
type: knowledge
tags:
  - ai
---
<iframe width="560" height="315" src="https://www.youtube.com/embed/eNIqz_noix8?si=aZxEho98aiFDlvfM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
# Backpropagation o retro-propagación
Es la forma como la [[Red Neuronal]] realiza su aprendizaje automático ajustando los pesos

<iframe width="560" height="315" src="https://www.youtube.com/embed/M5QHwkkHgAA?si=Vp81fuHutw25IafX" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

- [[libro perceptrons]] 
- [[Learning representations by back-propagating errors]]

## Autograd durante el entrenamiento
Hemos echado un breve vistazo a cómo funciona Autograd, pero ¿cómo se ve cuando se usa para el propósito previsto? Definíamos un modelo pequeño y examinamos cómo cambia después de un solo lote de entrenamiento. Primero, definimos algunas constantes, nuestro modelo y algunos sustitutos para entradas y salidas:
```python
BATCH_SIZE = 16
DIM_IN = 784
HIDDEN_SIZE = 256
DIM_OUT = 10

net = torch.nn.Sequential(torch.nn.Linear(DIM_IN, HIDDEN_SIZE),
						 torch.nn.ReLU(),
						 torch.nn.Linear(HIDDEN_SIZE, DIM_OUT))

# features aleatorias
some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
# etiquetas aleatorias
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = net
```

Fijémonos que no hizo falta agregar `requires_grad=True` a las capas de modelo, esto es por que la clase `torch.nn.Module` supone que siempre usaremos el gradiente para entrenar el modelo.

Sin embargo, al momento de iniciar los valores del modelo, el gradiente no se calcula, hasta que lo pidamos.

```python
print(model[2].weight[0][0:10]) # solo algunos son mostrados
print(model[2].weight.grad)
```

Veamos que ocurre ahora si entrenamos.
Consideraremos como función de perdida la distancia cuadrática media entre nuestra `prediction` y las etiquetas, `ideal_output` En este caso usaremos `SGD` como algoritmo de optimización.

 ```python
 optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
 prediction = model(some_input)
 loss = (ideal_output - prediction).pow(2).sum()
 print(loss)
 ```
 
Hasta que no llamemos `loss.backward()` los gradientes no se calculan.
```python
print(model[2].weight[0][0:10]) # solo algunos son mostrados
print(model[2].weight.grad)

loss.backward() # hay que pedir explicitamente calcular los gradientes
print(model[2].weight[0][0:10])
print(model[2].weight.grad[0][0:10])
```

Por ahora solo hemos calculado los gradientes, pero no los hemos usado para actualizar los pesos. Esto es porque debemos ejecutar `optimizer.step()`

```python
optimizer.step()
print(model[2].weight[0][0:10])
print(model[2].weight.grad[0][0:10])
```

Vemos ahora que los valores de `model[2]` han cambiado

Un detalle que no debemos ignorar es que debemos llamar a la función `optimizer.zero_grad()` después de llamar `optimizer.step()`. De no hacer esto cada vez que llamemos `loss.backward()` la suma de los gradientes se acumulará.

```python
print(model[2].weight.grad[0][0:10])

for i in range(0, 5):
	prediction = model(some_input)
	loss = (ideal_output - prediction).pow(2).sum()
	loss.backward()

print(model[2].weight.grad[0][0:10])
optimizer.zero_grad()

print(model[2].weight.grad[0][0:10])
```


### Contenido adicional: Más información sobre Autograd

En principio, ya conocíamos la noción de gradiente. Sabíamos que para una toma vectores m-dimensionales y devuelve un único valor (un escalar), $l=g\left(\vec{y}\right)$ existe el gradiente. Esto es un vector que nos dice como varía una función conforme cambian los valores del vector de entrada $\vec{y}$ 


$$v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}$$

En general, si tenemos una función que toma vectores n-dimensionales como entrada y tiene como salida vectores m-dimensionales, $\vec{y}=f(\vec{x})$, la idea de gradiente no permite abarcar todas las posibles variaciones. En este sentido se necesita una generalización de la idea de gradiente. Esta generalización es una matriz conocida como el *Jacobiano:*

$$\begin{align}J
     =
     \left(\begin{array}{ccc}
     \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
     \vdots & \ddots & \vdots\\
     \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
     \end{array}\right)\end{align}$$

Sin embargo, la función de pérdida nuestros modelos más sencillos son en realidad una combinación de las dos cosas. 

$$l=g\left(\vec{y}\right)$$
$$\vec{y}=f(\vec{x})$$
$$l=g\left(f(\vec{x})\right)$$

Puede demostrarse, sin embargo, que para obtener el gradiente de $l$, respecto de $\vec{x}$ solo debemos hacer una multiplicación matricial

$$\vec{\nabla_x} l=J^{T}\cdot v$$

$$\begin{align}J^{T}\cdot v=\left(\begin{array}{ccc}
   \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
   \vdots & \ddots & \vdots\\
   \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
   \end{array}\right)\left(\begin{array}{c}
   \frac{\partial l}{\partial y_{1}}\\
   \vdots\\
   \frac{\partial l}{\partial y_{m}}
   \end{array}\right)=\left(\begin{array}{c}
   \frac{\partial l}{\partial x_{1}}\\
   \vdots\\
   \frac{\partial l}{\partial x_{n}}
   \end{array}\right)\end{align}$$

Del mismo modo, a la salida de cada capa, tenemos un Jacobiano distinto. De tal manera que nuestro gradiente en realdiad tendra la forma:

$$\vec{\nabla_x} l=J_{1}^{T} J_{2}^{T} J_{3}^{T} J_{4}^{T}\cdot v$$


**``torch.autograd`` es la herramienta que computa todas estas dependencias por medio de productos matriciales** Además de guardar la relación entre cada salida y cada entrada de cada capa


# Referencias
1. [[Perceptron Multicapa]]
2. [[00 PyTorch fundamentals]]



