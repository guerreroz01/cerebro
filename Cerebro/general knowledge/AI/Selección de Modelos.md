# Selección de modelos
___
En el aprendizaje automático, generalmente seleccionamos nuestro modelo final después de evaluar varios modelos candidatos. Este proceso se llama _selección de modelo_. A veces los modelos sujetos a comparación son de naturaleza fundamentalmente diferente (por ejemplo, árboles de decisión frente a modelos lineales). En otras ocasiones, estamos comparando miembros de la misma clase de modelos que han sido entrenados con diferentes configuraciones de hiperparámetros.

Con los MLP, por ejemplo, es posible que deseemos compara modelos con diferentes números de capas ocultas diferentes números de unidades ocultas y varias opciones de funciones de activación aplicadas a cada capa oculta. Para determinar cuál es el mejor entre nuestros modelos candidatos, generalmente emplearemos un conjunto de datos de validación.

## Conjunto de datos de validación
____
En principio, no deberíamos tocar nuestro conjunto de prueba hasta que hayamos elegido todos nuestros hiperparámetros. Si utilizáramos los datos de prueba en el proceso de selección del modelo, existe el riesgo de que podamos sobreajustar los datos de prueba. Entonces estaríamos en serios problemas. Si sobreajustamos nuestros datos de entrenamiento, siempre existe la evaluación de los datos de prueba para mantenernos honestos. Pero si sobreajustamos los datos de prueba, ¿Cómo lo sabríamos?

Por lo tanto, nunca debemos confiar en los datos de prueba para la selección del modelo. Y, sin embargo, tampoco podemos confiar únicamente en los datos de entrenamiento para la selección del modelo porque no podemos estimar el error de generalización en los mismos datos que usamos para entrenar el modelo. 

En aplicaciones prácticas, la imagen se vuelve más turbia. Si bien, idealmente, solo tocaríamos los datos de prueba una vez, para evaluar el mejor modelo o para comparar una pequeña cantidad de modelos entre sí, los datos de prueba del mundo real rara vez se descartan después de un solo uso. Rara vez podemos permitirnos un nuevo conjunto de prueba para cada ronda de experimentos.

La práctica común para abordar este problema es dividir nuestros datos en tres maneras, incorporando un _conjunto de datos de validación_ (o conjunto de validación) además de los conjuntos de datos de entrenamiento y prueba.

![[Pasted image 20240128111741.png]]

Un buen ejemplo para distinguir entre conjunto de prueba y de validación es lo que hace la plataforma Kaggle en sus competencias de aprendizaje automático. En sus inicios, Kaggle era solamente una plataforma de concursos donde las empresas publican problemas y los participantes compiten para construir el mejor algoritmo, generalmente con premios en efectivo. La organización de los concursos consiste en:
1. El organizador debe separar su dataset en un conjunto de entrenamiento (que será publicado) y un conjunto de prueba (cuyas features serán publicadas, pero las etiquetas permanecerán ocultas).
2. Los participantes podrán descargar los datos de entrenamiento y deberán elegir un modelo para presentar en la competencia. Para eso, deberán llevar adelante una selección de modelos generando un conjunto de validación a partir de los datos de entrenamiento.
3. Una vez seleccionado el modelo que mejor funcione con los datos de validación, se alimenta dicho modelo con las features del conjunto de prueba para obtener las etiquetas de prueba predichas por el modelo. 
4. Se entregan las etiquetas de prueba predichas y el organizador las compara con las reales. El ganador es el modelo que menos errores haya cometido.

![[Pasted image 20240128112401.png]]

De esta manera, los conjuntos de prueba y validación están bien diferenciados. El primero se usa para elegir el mejor modelo y el segundo se usa para evaluar el modelo elegido con datos que nunca vio en el entrenamiento.

A menos que se indique explícitamente lo contrario, en los experimentos de este curso en realidad estamos trabajando con lo que correctamente debería llamarse datos de entrenamiento y datos de validación, sin verdaderos conjuntos de prueba. Por lo tanto, reportado en cada experimento es realmente un accuracy de validación y no un verdadero accuracy del conjunto de pruebas.

## K-folds cross-validation
___
Cuando los datos de entrenamiento son escasos, es posible que ni siquiera podamos permitirnos mantener suficientes datos para construir un conjunto de validación adecuado. Una solución popular a este problema es emplear $\text{K-folds croos-validation}$. Aquí, los datos de entrenamiento originales se dividen en $K$ subconjuntos que no se superponen. 

Luego, el entrenamiento y la validación del modelo se ejecutan $K$ veces, cada vez que entrenando en $K - 1$ subconjuntos y validando en un subconjunto diferente (el que no se usó para entrenar en esa ronda). Finalmente, los errores de entrenamiento y validación se estiman promediando los resultados de los experimentos de $K$.

![[Pasted image 20240128121631.png]]

```python
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader,ConcatDataset

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
```

### Model

Definamos una red neuronal simple para el conjunto de datos MNIST.

```python
INPUT = 28 * 28 # 28 POR 28 PIXELES
OUTPUT = 10 # 10 CLASES
# TODO
HIDDEN1 = 512
HIDDEN2 = 128

net = nn.Sequential(nn.Flatten(),
				   nn.Linear(INPUT, HIDDEN1),
				   nn.ReLU(),
				   nn.Linear(HIDDEN1, HIDDEN2),
				   nn.ReLU(),
				   nn.Linear(HIDDEN2, OUTPUT))
```

### Función para reiniciar pesos

Necesitamos restablecer los pesos del modelo para que cada fold de cross validation comience desde un estado inicial aleatorio y no aprenda de los folds anteriores. Podemos llamar a `reset_weights()` en todos los módulos hijos.

```python
def reset_weights(m):
	if type(m) == nn.Linear:
		nn.init.normal_(m.weights, std=0.01)
```

Modificamos ligeramente los pipelines de entrenamiento para que sea más ordenado... Todas las líneas para calcular la pérdida y mejorar los parámetros los ponemos en la función train y todas las que se encargan de calcular el accuracy, en la función test.

```python
def train(fold, model, device, loss, train_loader, optimizer, epoch):
	for barch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		l = loss(model(data), target).mean()
		l.backward()
		optimizer.step()
		if batch_idx % 500 == 0:
			print('Train Fold/Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(fold,epoch, batch_idx * len(data), len(train_loader.sampler.indices),100. * batch_idx / len(train_loader), l.item()/len(target)))
```

```python
def accuracy(y_hat, y):
	"""Compute the number of correct predictions."""
	if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
		y_hat = y_hat.argmax(axis=1)
	cmp = y_hat.type(y.dtype) == y
	return float(cmp.type(y.dtype).sum())

def test_accuracy(fold, model, loss, device, test_loader):
	# insete su código aquí
	TestAcc = 0.0
	N = 0
	for X, y in test_loader:
		X, y = X.to(device), y.to(device)
		N += y.numel()
		TestAcc += accuracy(model(X), y)
		print('\nTest set for fold {}: Accuracy: {}/{} ({:.0f}%)\n'.format(fold, TestAcc, N,(100. * TestAcc) / N))
	return TestAcc / N
```

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
```

### Dataset
___
Necesitamos concatenar las partes de entrenamiento y prueba del dataset MNIST, que usaremos para entrenar el modelo. Hacer K-fold implica que nosotros mismo generemos las divisiones, por lo que no queremos que PyTorch lo haga por nosotros.

```python
transform=transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((o.1307,), (0.3081))
])

dataset1 = datasets.MNIST('../data', train=True, download=True,
						  transform=transform)

dataset2 = datasets.MNIST('../data', train=False,
						  transform=transmform)
```

```python
dataset=ConcatDataset([dataset1,dataset2])
```


### Clase KFold
___
KFold es una clase de la librería sklearn que nos puede ayudar a hacer cross validation. Para eso debemos instanciar el objeto kfold indicando la cantidad de folds que queremos en el atributo `n_splits` del constructor.

```python
kfold=KFold(n_splits=5, shuffle=True)
```

La clase KFold tiene un método llamado `split()` que es un iterador que recibe el dataset a separar y devuelve un tupla con dos listas de índices. La primera es la lista de índices de entrenamiento y la segunda es la lista de índices de testeo de ese fold.

```python
for train_idx, test_idx in kfold.split(dataset):
	print('train indices', len(train_idx), train_idx)
	print('test indices', len(test_idx), test_idx)
```

Ahora podemos generar los folds y entrenar nuestro modelo. Lo vamos a hacer definiendo un loop que itere sobre los folds especificando la lista de identificadores de los ejemplos de entrenamiento y validación para eso fold en particular.

Dentro del loop hacemos un print del id del fold. Después, entrenamos muestreando los elementos de train y test con un SubsetRandomSampler. A esta clase se le puede pasar una lista con los índices de los elementos que debe muestrear del dataset.

```python
model = net1.to(device)
model.apply(reset_weights)
loss = torch.nn.CrossEntrophyLoss(reduction='none')
optimizer = optim.Adadelta(model.parameters())
```

```python
batch_size=32
folds=5
epochs=5
acc = []
for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset)):
	print('------------fold no---------{}----------------------'.format(fold))
	train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
	test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
	testloader = torch.utils.data.DataLoader(dataset,
batch_size=batch_size, sampler=test_subsampler)

	model.apply(reset_weights)

	fold_acc = 0
	for epoch in range(1, epochs + 1):
		train(fold, model, device, loss, trainloader, optimizer, epoch)
		fold_acc = test_accuracy(fold,model, loss, device, testloader)
		acc.append(fold_acc)
```

```python
print('El accuracy de cada fold es el siguiente {} y el accuracy promedio del modelo es {}'.format(
                acc, np.array(acc).mean()))
```

# Referencia
___
1. [[Perceptron Multicapa]]
2. [[00 PyTorch fundamentals]]
3. [[Deep Learning]]