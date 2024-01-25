# Multilayer Perceptron
Hasta aquí hemos trabajado con modelos lineales o modelos de una sola capa. Es fácil entender porque estos modelos simples pueden fallar. Por ejemplo, hay fenómenos que pueden tener ciclos o frecuencias asociadas. O pueden existir situaciones donde una de nuestras categorías caiga en un rango acotado de valores (por ejemplo cantidad saludable de azúcar en sangre).

Esto nos da a entender que debemos considerar más que solo modelos lineales

## Agregando capas ocultas
![[Pasted image 20240125230147.png]]

Sin embargo este modelo sigue siendo lineal:
$$
\begin{aligned}
\mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
\mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}. \\
\mathbf{O} & = [\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}]\mathbf{W}^{(2)} + \mathbf{b}^{(2)} \\
\mathbf{O} & = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)} \\
\mathbf{O} & = \mathbf{X} \mathbf{W'} + \mathbf{b'} + \mathbf{b}^{(2)}\\
\mathbf{O} & = \mathbf{X} \mathbf{W'} + \mathbf{b''}
\end{aligned}

$$


Si queremos un modelo más general, debemos usar algo más que solo capas lineales.

Para esto introducimos **_funciones de activación_**.
$$
\begin{aligned}
\mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
\mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$
## Funciones de activación
Lo que hacen las funciones de activación [[Función de activación]] es: a la salida de nuestra capa se agrega algo, que lo que hace es agregarle no linealidad a nuestro problema (deforma el resultado de la salida).

```python
%matplotlib inline
import torch
import matplotlib.pyplot as plt
import numpy as np
```

A continuación graficaremos algunas funciones de activación y sus respectivas derivadas

### ReLU 
$$\operatorname{ReLU}(x) = \max(0, x)$$
```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
plt.plot(x.detach(), y.detach())
```
![[Pasted image 20240125231840.png]]
```python
y.backward(torch.ones_like(x), retain_graph=True)
plt.plot(x.detach(), x.grad)
```
![[Pasted image 20240125231917.png]]

Existen varias alternativas similares a ReLU, sin embargo, queremos destacar a pReLU. A diferencia de ReLU, pReLU no descarta el gradiente a la izquierda. Además, el peso que se le otorgó al gradiente a la izquierda es parte de los parámetros del modelo.

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### Sigmoidea
$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

continuar aquí https://youtu.be/H5ptZkbzoVg?si=cVumyI86qLLimHEZ&t=779