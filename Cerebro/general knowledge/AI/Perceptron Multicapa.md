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

Por mucho tiempo se usó la función sigmoideo porque se trató de emular el funcionamiento de una neurona natural 

```python
y = torch.sigmoid(x)
plt.plot(x.detach(), y.detach())
```
![[Pasted image 20240126110203.png]]

```python
# Clear out previous gradients
x.grad.data.zero()
y.backward(torch.ones_like(x), retain_graph=True)
plt.plot(x.detach(), x.grad
```
![[Pasted image 20240126110357.png]]

### Tangente hiperbólica
$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$
$$\operatorname{tanh}(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}.$$

```python
y = torch.tanh(x)
plt.plot(x.detach(), y.detach())
```
![[Pasted image 20240126111026.png]]
```python
# Clear out previous gradients.
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
plt.plot(x.detach(), x.grad)
```
![[Pasted image 20240126111053.png]]

## Implementación del Perceptrón Multicapa desde 0

```python
import torch
from torch import nn
import torchvision
from IPython import display
from torchvision import tansforms
from torch.utils import data
```

```python
# Ejemplo de dataloader para Fashion MNIST
def load_data_fashion_mnist(batch_size, resize=None):
	trans = [transforms.ToTensor()]
	if resize:
		trans.insert(0, transforms.Resize(resize))
	trans = transforms.Compose(trans)
	mnist_train = torchvision.datasets.FashionMNIST(
		root="../data", train=True, transform=trans, download=True)
	mnist_test = torchvision.datasets.FashionMNIST(
		root="../data", train=False, transform=trans, download=True)
	return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=1), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=1))
```

```python
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
```

#### Inicialización de parámetros
```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
	num_inputs, num_hiddens, requires_grad=True, * 0.01
))
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
	num_hiddens, num_outputs, requires_grad=True, * 0.01
))
b2 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
params = [W1, b1, W2, b2]
```

#### Función de activación
```python
def relu(X):
	a = torch.zeros_like(X)
	retur torch.max(X, a)
	```

#### Modelo
```python
def net(X):
	X = X.reshape((-1, num_inputs))
	H = relu(X @ W1 + b1)
	return (H @ W2 + b2)
```

#### Función de pérdida
 ```python
 loss = nn.CrossEntropyLoss(reduction='none')
 ```

Recordemos que vamos a usar de nuevo la exactitud
```python
def accuracy(y_hat, y):
	""" Compute the number of correct predictions. """
	if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
		y_hat = y_hat.argmax(axis=1)
	cmp = y_hat.type(y.dtype) == y
	return float(cmp.type(y.dtype).sum())
```

#### Entrenamiento
```python
num_epochs, lr = 10, 0.1
updater = torch.optim.DGD(params, lr=lr)
```

```python
for epoch in range(num_epochs):
	L = 0.0
	N = 0
	Acc = 0.0
	TestAcc = 0.0
	TestN = 0
	for X, y in train_iter:
		l = loss(net(X), y)
		updater.zero_grad()
		l.mean().backward()
		updater.step()
		L += l.sum()
		# Aca el número de ejemplos
		N += l.numel()
		# Aca calculamos exactitud
		Acc += accuracy(net(X), y)
	for X, y in test_iter:
		TestN += y.numel()
		TestAcc += accuracy(net(X), y)
	print(f'ecpoch {epoch + 1}, loss {(L/N):f} \, train accuracy {(Acc/N):f}, test accuracy {(TestAcc/TestN):f}')
	```

### Implementación concisa con PyTorch
```python
import torch
from torch import nn
import torchvision
from IPython import display
from torchvision import transforms
from torch.utils import data

net = nn.Sequential(nn.Flatten(),
					nn.Linear(784, 256),
					nn.ReLU(),
					nn.Linear(256, 10))

def init_weights(m):
	if type(m) == nn.Linear:
		nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

#Ejemplo de dataloader para FAshin MNIST
def load_data_fashion_mnist(batch_size, resize=None):

	trans = [transforms.ToTensor()]

	if resize:
		trans.insert(0, transforms.Resize(resize))
	trans = transforms.Compose(trans)
	mnist_train = torchvision.datasets.FashionMNIST(
		root="../data", train=True, transform=trans, download=True)
	mnist_test = torchvision.datasets.FashionMNIST(
		root="../data", train=False, transform=trans, download=True)
	return (data.DataLoader(mnist_train, batch_size, shuffle=True,
							num_workers=1),
			data.DataLoader(mnist_test, batch_size, shuffle=False,
							num_workers=1))

def accuracy(y_hat, y):
	"""Compute the number of correct predictions."""
	if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
		y_hat = y_hat.argmax(axis=1)
	cmp = y_hat.type(y.dtype) == y
	return float(cmp.type(y.dtype).sum())

train_iter, test_iter = load_data_fashion_mnist(batch_size)

for epoch in range(num_epochs):
	L = 0.0
	N = 0
	Acc = 0.0
	TestAcc = 0.0
	TestN = 0
	for X, y in train_iter:
		l = loss(net(X) ,y)
		trainer.zero_grad()
		l.mean().backward()
		trainer.step()
		L += l.sum()
		N += l.numel()
		Acc += accuracy(net(X), y)
	for X, y in test_iter:
		TestN += y.numel()
		TestAcc += accuracy(net(X), y)
	print(f'epoch {epoch + 1}, loss {(L/N):f}\
			, train accuracy {(Acc/N):f}, test accuracy {(TestAcc/TestN):f}')
```

