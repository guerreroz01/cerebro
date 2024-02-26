---
sticker: emoji//1f6a7
type: knowledge
tags:
  - ai
  - pytorch
---
<iframe width="900" src="https://www.youtube.com/embed/Z_ikDlimN6A?si=FyN9GUMcHYR5Zuim" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
# PyTorch fundamentales
___ 
## Deep learning:
Machine learning convierte los datos en números y luego busca patrones en esos números.

¿Como consigue esos patrones?
Con Matemáticas.
![[Pasted image 20231129165812.png]]
## ¿Por qué usar machine learning (o deep learning)?
Porque se pueden resolver problemas más complejos por ejemplo; un coche que se conduce solo.

>[!abstract] Yashaswi Kulshreshtha
>I think you can use ML for literally anything as long as you can convert it into numbers and program it to find patterns. Literally it could be anything any input or output from the universe.

## ¿Para que es bueno Deep Learning 🤖?
1. Problemas con una lista muy larga de reglas.
2. Cuando las variables de entorno cambian constantemente, deep learning se puede adaptar (aprender) en los nuevos escenarios.
3. Cuando se tienen Data sets muy grandes.

## ¿Para qué no es bueno usar Deep Learning ❌?
1. Cuando se necesita explicación: esto es porque los patrones que usa el Deep Learning para aprender, tipicamente no pueden ser interpretados por los humanos.
2. Cuando el approach tradicional es la mejor opción: no te compliques si se puede resolver el problema con unas pocas reglas.
3. Cuando los errores son inaceptables: esto es porque la salida de Deep Learning a veces son impredecibles.
4. Cuando no se tiene suficiente data para entrenar el modelo para que pueda dar buenos resultados.

>[!warning] Cuando usar Deep Learning o Machine Learning
>Machine Learning funciona mejor cuando la data está estructurada.
>Deep Learning es mejor cuando la data no está estructurada.

## Tipos de aprendizajes:
1. **Aprendizaje supervisado**: un ejemplo sería cuando la data está etiquetada como en una foto con metadata.
2. **Aprendizaje no supervisado** es decir el modelo aprende solo: la data no está etiquetada así que el modelo debe encontrar los patrones por si solo.
3. **Aprendizaje transferido**: es muy importante porque se puede traspasar lo que aprendió un modelo a otro modelo para que tome ese conocimiento como los fundamentos de su aprendizaje.
4. **Aprendizaje reforzado**: es cuando se le da una pista al modelo en cada etapa de su aprendizaje para que vaya obteniendo feedback.

## PyTorch 
Es un framework para investigación en el campo del deep learning, se puede escribir código de deep learning de manera rápida en python y correrlo en cualquier GPU, se puede acceder a muchos modelos pre construidos de deep learning y tiene todo el stack para el desarrollo de software hasta el despliegue en la nube.

### ¿Que es un Tensor?
Los tensors es la representación numérica de los datos y la salida numérica de la capa de salida antes de ser transformada para poder ser entendida por los humanos.
![[Pasted image 20231129180504.png]]
<iframe width="560" height="315" src="https://www.youtube.com/embed/f5liqUk0ZTw?si=2qk5U1FRX3Zdh172" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Los tensores son la abstracción de datos central en PyTorch. `torch.tensor`

Lo primero es lo primero, importemos el módulo PyTorch. También agregaremos el módulo `math` de Python para facilitar algunos de los ejemplos.

```python
import torch
import math
```

#### Crear tensores
La forma más sencilla de crear un tensor es con la llamada `torch.empy()`
```python
x = torch.empty(3, 4) # donde los valores serán las dimensiones, el primer valor son las filas y el segundo las columnas
```

Desempaquemos lo que acabamos de hacer:
- Creamos un tensor utilizando uno de los numerosos métodos constructores provistos por `torch`.
- El tensor en sí es bidimensional, tiene 3 filas y 4 columnas.
- El tipo de objeto devuelto es `torch.Tensor`, que es un alias para `torch.FloatTensor`; por defecto, los tensores de PyTorch son poblados con números de punto flotante de 32 bits.
- Probablemente verá algunos valores de aspecto aleatorio al imprimir su tensor. La llamada `torch.empty()` asigna memoria para el tensor, pero no lo inicializa con ningún valor, por lo que lo que está viendo es lo que estaba en la memoria en el momento de la asignación.

Una breve nota sobre los tensores y su número de dimensiones y terminología
- A veces verás un tensor unidimensional llamado **vector**
- Del mismo modo, un tensor bidimensional a menudo se denomina **matriz**
- Cualquier cosa con más de dos dimensiones generalmente es solo llamado tensor.

La mayoría de las veces, querrá inicializar su tensor con algún valor. Los casos comunes son todos ceros, todos unos o valores aleatorios, y el módulo `torch` proporciona métodos constructores para todos estos.

```python
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)

torch.manual_seed(1729) # para que siempre de un valor aleatorio predecible
random = torch.rand(2, 3)
```

#### Tensores Aleatorios y Semillas 
Hablando del tensor aleatorio, ¿notaste la llamada a `torch.manual_seed()` inmediatamente anterior? inicializar tensores, como los pesos de aprendizaje de un modelo, con valores aleatorios es común pero hay momentos, especialmente en entornos de investigación, en los que querrá cierta seguridad de la reproducibilidad de sus resultados. Asignar manualmente la semilla de su generador de números aleatorios es la forma de hacer esto. Miremos más cerca:
```python
torch.manual_seed(1729)
random1 = torch.rand(2,3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2,3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)
```


#### Formas de los Tensores
A menudo, para poder realizar operaciones en dos o más tensores, estos tendrán que tener la misma _**forma**_, es decir tener el mismo número de dimensiones y el mismo número de celdas en cada dimensión. Para garantizar eso existen los métodos `torch.*_like()`
```python
x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)

# Todos tienen las mismas dimensiones pero con sus valores correspondientes
```

Lo primero nuevo en la celda de código de arriba es el uso del atributo `.shape` de todo tensor. Este atributo contiene una lista con el tamaño de cada dimensión de un tensor, en nuestro caso, x es un tensor tridimensional con forma 2 x 2 x 3.

La última forma de crear un tensor es especificar sus datos directamente desde una colección de PyTorch:

```python
some_constants = torch.tensor([[3.14332, 2.732342], [1.4374973, 0.00834]])
print(some_constants)

some_integers = torch.tensor((2, 3, 4, 6, 1, 9, 10, 30))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9])) # No importa al constructor si se pasan tuplas o listas, lo unico que toma en cuenta es el anidamiento.
print(more_integers)
```

Usar `torch.tensor()` es la forma más sencilla de crear un tensor si ya tiene datos en una tupla o lista de Python. Como se muestra anterior, el anidamiento de las colecciones dará como resultado un tensor multidimensional.

#### Tipos de datos de un Tensor
Establecer el tipo de datos de un tensor es posible de dos maneras:
```python
a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20
print(b)

c = b.to(torch.int32)
print(c)
```

La forma más sencilla de establecer el tipo de datos subyacente de un tensor es con un argumento opcional en el momento de la creación. En la primera línea de la celda de arriba, configuramos `dtype=torch.int16` para el tensor `a`. podemos ver que está lleno de `1` en lugar de `1.` - Python indica que este es un tipo entero en lugar de un punto flotante.

Otra cosa a tener en cuenta sobre la impresión de `a` es que, a diferencia de cuando dejamos `dtype` como predeterminado (com flotante de 32 bits), imprimiendo el tensor también especifica su `dtype`.

Es posible que también haya notado que pasamos de especificar la forma del tensor como una serie de argumentos enteros, a agrupar esos argumentos en un `tupla`. Esto no es estrictamente necesario: PyTorch tomará una serie de argumentos enteros iniciales sin etiquetar como la forma del tensor, pero al agregar los argumentos opcionales, ponerlos como tupla puede hacer que su intención sea más legible,

La otra forma de establecer el tipo de datos es con el método `.to()` . En la celda de arriba, creamos un tensor de punto flotante aleatorio `b` de la manera habitual. A continuación, creamos `c` convirtiendo `b` en un tensor entero de 32 bits con el método `to()` . Tenga en cuenta que `c` contiene todos los mismos valores como `b`, pero truncados a enteros.

Los tipos de datos disponibles incluyen:
- `torch.bool`
- `torch.int8`
- `torch.uint8`
- `torch.int16`
- `torch.int32`
- `torch.int64`
- `torch.half`
- `torch.float`
- `torch.double`
- `torch.bfloat`

#### Matemáticas y lógica con tensores PyTorch
Ahora que conoces algunas de las formas de crear un tensor... ¿qué puedes hacer con ellos?

Veamos primero la aritmética básica y cómo interactúan los tensores con escalares simples:
```python
ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 3) + 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5
# cuando los tensores se operan con escalares las operaciones se hacen uno a uno
```

Como puedes ver arriba, las operaciones aritméticas entre tensores y escalares, como suma, resta, multiplicación, división y exponenciación se aplican elemento a elemento dentro del tensor. Dado que la salida de tal operación será un tensor, puedes encadenarlos junto con las reglas usuales de precedencia de operadores, como en la línea donde creamos `threes`.

Las operaciones similares entre dos tensores también se comportan intuitivamente:
```python
powers2 = two ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)
```

#### Broadcasting de Tensores
En el caso general, no se puede operar con tensores de diferente forma de esta manera, incluso en un caso como el de la celda anterior donde los tensores tienen un idéntico número de elementos.

>[!tip] Nota
>Si está familiarizado con el broadcasting de NumPy, aquí se aplican las mismas reglas. La excepción a la regla de las mismas formas es el **broadcasting de tensores**

¿Cuál es el truco aquí? ¿Cómo es que podemos multiplicar un tensor de 2x4 por un tensor 1x4?

El broadcasting es una forma de realizar una operación entre tensores que tienen similitudes en sus formas. En el ejemplo anterior, el tensor de cuatro columnas, el de una fila, se multiplica por _ambas filas_ del tensor de cuatro columnas y dos filas.

Esta es una operación importante en Deep Learning. El ejemplo común es multiplicar un tensor de pesos de aprendizaje por un _lote_ de tensores de entrada, aplicando la operación a cada instancia en el lote por separado, y devolviendo un tensor de forma idéntica, al igual que nuestro (2, 4) * (1, 4) ejemplo anterior devolvió un  tensor de forma (2, 4).

Las reglas para el broadcasting son:
- Cada tensor debe tener al menos una dimensión - no hay tensores vacíos.
- Comparando los tamaños de las dimensiones de los dos tensores, _yendo del último al primero:_
	- Cada dimensión debe ser igual, o
	- Una de las dimensiones debe ser de tamaño 1, o
	- La dimensión no existe en uno de los tensores
Los tensores de forma idéntica, por supuesto, son trivialmente "broadcasteables", como viste antes.

Aquí hay algunos ejemplos de situaciones que respetan las reglas anteriores y permiten el broadcasting.
```python
a =     torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 3era & 2da dims identicas a las de a, dim 1 ausente
c = a * torch.rand(   3, 1) # 3era dim=1, 2da dim identica a la de a

d = a * torch.rand(   1, 2) # 3era dim identica a las de a, 2da dim=1
```


### Más matemáticas con tensores
Los tensores PyTorch tienen más de trescientas operaciones que se pueden realizar en ellos.

Aquí hay una pequeña muestra de algunas de las principales categorías de operaciones:
```python
a = torch.rand(2, 4) * 2 - 1
print('Common funtions:')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

# trigonometric functions and their inverses
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print('\nSine and arcsine:')
print(angles)
print(sines)
print(inverses)

# bitwise operations
print('\nBitwise XOR:')
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

# comparisons:
print('\Broadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2) # many comparison ops support broadcasting!
print(torch.eq(d, e)) # returns a tensor of type bool

# reductions:
print('\nReduction ops:')
print(torch.max(d)) # returns a single-element tensor
print(torch.max(d).item()) # extracts the value from the returned tensor
print(torch.mean(d)) # average
print(torch.std(d)) # standard deviation
print(torch.prod(d)) # product of all numbers
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2]))) # filter unique elements

# vector and linear algebra operations
v1 = torch.tensor([1., 0., 0.,]) # x unit vector
v2 = torch.tensor([0., 1., 0.]) # y unit vector
m1 = torch.rand(2, 2) # random matrix
m2 = torch.tensor([[3., 0.], [0., 3.]]) # three times identity matrix

print('\nVectors & Matrices:')
print(torch.cross(v2, v1)) # negative of z unit vector (v1 x v2 == -v2 x v1)
print(m1)
m3 = torch.matmul(m1, m2)
print(m3) # 3 times m1
print(torch.cvd(m3)) # singular value decomposition
```

#### Alteración de tensores en su lugar
La mayoría de las operaciones binarias entre tensores devolverán un tercer tensor nuevo. Cuando decimos `c = a * b` (donde `a` y `b` son tensores), el nuevo tensor `c` ocupará una región de memoria distinta de los otros tensores.

Sin embargo, hay ocasiones en las que es posible que desee alterar un tensor en su lugar: por ejemplo, si está haciendo un cálculo por elementos en el que puede descartar valores intermedios. Para esto, la mayoría de las funciones matemáticas tienen un versión con un guión bajo adjunto (`_`) que alterará un tensor en su lugar.

Por ejemplo:
```python
a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('a: ')
print(a)
print(torch.sin(a)) # esta operación crea un nuevo tensor en la memoria
print(a) # a no ha cambiado

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('\nb: ')
print(b)
print(torch.sin_(b)) # note el guión bajo
print(b) b ha cambiado
```

Para las operaciones aritméticas existen funciones que se comportan de manera similar.

```python
a = torch.ones(2, 2)
b = torch.rand(2, 2)

print('Before:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b)
```

Tenga en cuenta que estas funciones aritméticas is situ son métodos del objeto `torch.Tensor`, no adjunto al módulo `torch` como muchos otras funciones (por ejemplo, `torch.sin()`). Como puedes ver desde `a.add_(b)`, _el tensor de llamada es el que se cambia en lugar._

Existe otra opción para colocar el resultado de un cálculo en un tensor asignado existente. Muchos de los métodos y funciones que hemos visto hasta ahora, ¡Incluidos los métodos constructores!, tienen un argumento `out` que le permite especificar un tensor para recibir la salida. Si el tensor `out` es de la forma correcta y `dtype` correcto, esto puede suceder sin una nueva asignación de memoria:

```python
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)  # el contenido de c ha cambiado

assert c is d # se fija si c & d son el mismo objeto, no que solo contienen los mismos valores

assert id(c), old_id # se asegura que el nuevo c sea el mismo que el viejo

torch.rand(2, 2, out=c) # funciona también para constructores
print(c) # c ha cambiado nuevamente
assert id(c), old_id # todavía es el mismo objeto

```

#### Copiando tensores
Como cualquier objeto en Python, asignar un tensor a una variable convierte a la variable en una `etiqueta` del tensor y no la copia. Por ejemplo:

```python
a = torch.ones(2, 2)
b = a

a[0][1] = 561 # al cambiar a
print(b)  # ... b también se altera

```

Pero, ¿qué sucede si deseas una copia separada de los datos para trabajar? El método `clone()` está ahí para ti:
```python
a = torch.ones(2, 2)
b = a.clone()

assert b is nor a # diferentes objetos en memoria...
print(torch.eq(a, b)) # ... pero todavía son el mismo contenido!

a[0][1] = 561 # a cambia...
print(b)  # ...pero b todavía son puros unos
```

#### Manipulación de la forma del tensor
A veces, necesitarás cambiar la forma de tu tensor. A continuación, veremos algunos casos comunes y cómo manejarlos.

#### Cambiar el número de dimensiones
Un caso en el que podría necesitar cambiar la cantidad de dimensiones es para una sola instancia como entrada a su modelo. Los modelos de PyTorch generalmente esperan `lotes` de entrada.

Por ejemplo, imagine tener un modelo que funcione con imágenes de 3 x 226 x 226, un cuadrado de 226 pixeles con 3 canales de color. Cuando lo cargues y lo transformes, obtendrás un tensor de forma `(3, 226, 226)`. Sin embargo, su modelo espera una entrada de forma `(N, 3, 226, 226)` donde `N` es el número de imágenes en el lote. Entonces ¿cómo se hace un lode de uno?

```python
a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)

```

El método `unsqueeze()` agrega una dimensión de extensión 1. `unsqueeze(0)` lo agrega como una nueva dimensión cero - ¡ahora tienes un lote de uno!

Estamos aprovechando el hecho de que cualquier dimensión de extensión 1 `no`cambia el número de elementos en el tensor.

```python
c = torch.rand(1, 1, 1, 1, 1)
print(c)
```

Continuando con el ejemplo anterior, digamos que la salida del modelo es un vector de 20 elementos para cada entrada. Entonces esperaría que la salida tuviera la forma (N, 20), donde `N` es el número de instancias en el lote de entrada. Eso significa que para nuestro lote de entrada única, obtendremos una salida de forma (1, 20).

¿Qué sucede si desea realizar un cálculo `no por lotes` con esa salida, algo que solo espera un vector de 20 elementos?
```python
a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)
```

Puede ver en las formas que nuestro tensor bidimensional ahora es 1-dimensional, y si miras de cerca la salida de la celda de arriba verás que imprimir `a` muestra un conjunto "extra" de corchetes [ ] debido a que tienen una dimensión adicional.

Solo puede hacer `squeeze()` sobre las dimensiones de tamaño 1. Vea arriba, donde tratamos de comprimir una dimensión de tamaño 2 en c, y terminamos recuperando la misma forma con la que comenzamos. Las llamadas a `squeeze()` y `unsqueeze()` solo pueden actuar en dimensiones de tamaño 1 porque, de lo contrario, cambiaría el número de elementos en el tensor.

Otro lugar en el podrías usar `unsqueeze()` es para facilitar el broadcasting. Recuerde el ejemplo anterior donde teníamos el siguiente código:
```python
a = torch.ones(4, 3, 2)
c = a * torch.rand(   3, 1) # 3rd dim=1, 2nd dim identical to a
print(c)

```

A veces querrá cambiar la forma de un tensor de forma más radical, conservando al mismo tiempo la cantidad de elementos y su contenido.

`reshape()` hará esto por ti, siempre que las dimensiones que solicites produzcan el mismo número de elementos que tiene el tensor de entrada:
```python
output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# can also call it as a method on the torch module:
print(torch.reshape(output3d, (6 * 20 * 20)).shape)
```

El argumento `(6 * 20 * 20,)` en la línea de la celda anterior se debe a que PyTorch espera una **tupla** al especificar una forma de tensor, pero cuando la forma es el primer argumento de un método, nos permite hacer trampa y simplemente usar una serie de números enteros. Aquí, tuvimos que agregar los paréntesis y la coma para convencer al método de que se trata realmente de una tupla de un elemento.

Cuando pueda, `reshape()` devolverá una `vista` del tensor a ser cambiado, es decir, un objeto tensor separado que mira la misma región subyacente de la memoria. _Esto es importante:_ Eso significa que cualquier cambio realizado en el tensor fuente se reflejará en la vista de ese tensor, a menos que le hagas `clone()`

#### Puente con NumPy
En la sección anterior sobre broadcasting, se mencionó que semántica de broadcasting de PyTorch es compatible con la de NumPy, pero la afinidad entre PyTorch y NumPy es aún más profunda que eso.

Si tiene código científico o de ML pre-existente con datos almacenados en NumPy ndarrays, es posible que desee expresar esos mismos datos como tensores PyTorch, ya sea para aprovechar la aceleración GPU de PyTorch o sus abstracciones eficientes para construir modelos neuronales.

### Datasets & DataLoaders
El código para procesar muestras de datos puede complicarse y ser difícil de mantener. Idealmente, queremos que nuestro código de conjunto de datos esté desacoplado de nuestro código de entrenamiento del modelo para una mejor legibilidad y modularidad. PyTorch proporciona dos primitivas de datos: `torch.utils.data.DataLoader` y `torch.utils.data.Dataset` que nos permiten usar datasets precargados, así como nuestros propios datos. `Dataset` almacena las muestras y sus etiquetas correspondientes, y `DataLoader` envuelve un iterable alrededor del `Dataset` para facilitar el acceso a las muestras.

PyTorch proporciona una serie de datasets precargados (como FashionMNIST) que son una subclase de `torch.utils.data.Dataset` e implementan funciones específicas para los datos en particular. Se pueden usar para crear prototipos y comparar su modelo.

#### Cargar un Dataset
Este es un ejemplo de cómo cargar el conjunto de datos `Fashion-MNIST <https://research.zalando.com/project/fashion_mnist/fashion_mnist/>` de TorchVision. Fashion-MNIST es un conjunto de datos de imágenes de artículos de Zalando que consta de 60000 ejemplos de entrenamiento y de 10000 ejemplos de prueba. Cada ejemplo comprende una imagen en escala de grises de 28 x 28 y una etiqueta asociada de una de las 10 clases.

Cargamos el conjunto de datos `FashionMNIST <https://pythorch.org/vision/stable/datasets.html#fashion-mnist>` con los siguientes parámetros:
- `root` es la ruta donde se almacenan los datos del entrenamiento/prueba,
- `train` especifica si es un dataset de entrenamiento o prueba,
- `download=True` descarga los datos de Internet si no están disponibles en `root`.
- `transform` y `target_tansform` especifican las transformaciones de características y etiquetas.

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
	root='data',
	train=True,
	download=True,
	transform=ToTensor()
)

test_data = datasets.FashionMNIST(
	root='data',
	train=False,
	download=True,
	transform=ToTensor()
)
```

#### Iterar y Visualizar el Dataset
Podemos indexar `Datasets` manualmente como una lista: `training_data[index]`. Usamos `matplotlib` para visualizar algunas imágenes de nuestros datos de entrenamiento.

```python

labels_map = {
	0: 'T-Shirt',
	1: 'Trouser',
	2: 'Pullover',
	3: 'Dress',
	4: 'Coat',
	5: 'Sandal',
	6: 'Shirt',
	7: 'Sneaker',
	8: 'Bag',
	9: 'Ankle Boot'
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
	sample_idx = torch.randint(len(training_data), size=(1,)).item()
	img, label = training_data[sample_idx]
	figure.add_subplot(rows, cols, i)
	plt.title(labels_map[label])
	plt.axis('off')
	plt.imshow(img.squeeze(), cmap='gray')
plt.show()
```

#### Creación de un Dataset personalizado para nuestros archivos
Una clase Dataset personalizada debe implementar tres funciones: `__init__, __len__` y `__getitem__`. Echemos un vistazo a esta implementación; las imágenes de FashionMNIST se almacenan en un directorio `img_dir`, y sus etiquetas se almacenan por separado en un archivo CSV `annotations_file`.

En las siguientes secciones, desglosaremos lo que sucede en cada una de estas funciones.

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
	self.img_labels = pd.read_csv(annotations_file)
	self.img_dir = img_dir
	self.transform = transform
	self.target_transform = target_tansform

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
		image = read_image(img_path)
		label = self.img_labels.iloc[idx, 1]
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image, label
```

#### `__init__`
La función **init** se ejecuta una vez al instanciar el objeto Dataset. Inicializamos el directorio que contiene las imágenes, el annotations_file y ambas transformaciones (tratadas con más detalles en la siguiente sección).

El archivo etiquetas.csv se ve así:
```
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```
```python
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
	self.img_labels = pd.read_csv(annotations_file)
	self.img_dir = img_dir
	self.transform = transform
	self.target_transform = target_tansform
```
#### `__len__`
la función **len** devuelve el número de muestras en nuestro dataset:
Ejemplo:
```python
def __len__(self):
	return len(self.img_labels)
```

#### `__getitem__`
La función **getitem** carga y devuelve una muestra del dataset en el índice dado `idx`. Según el índice, identifica la ubicación de la imagen en el disco, la convierte en un tensor usando `read_image`, recupera la etiqueta correspondiente de los datos `csv`en `self.img_labels`, llama a las funciones de transformación en ellos (si corresponde) y devuelve la imagen del tensor y la etiqueta correspondiente en una tupla.

```python

def __getitem__(self, idx):
	img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
	image = read_image(img_path)
	label = self.img_labels.iloc[idx, 1]
	if self.transform:
		image = self.transform(image)
	if self.target_transform:
		label = self.target_transform(label)
	return image, label
```

#### Preparando nuestros datos para el entrenamiento con DataLoaders
El `Dataset` recupera las características y etiquetas de nuestro conjunto de datos en una muestra a la vez, Mientras entrenamos un modelo, normalmente queremos pasar muestras en "minilotes", mezclar los datos en cada época para reducir el sobreajuste del modelo y usar el `multiprocesamiento` de Python para acelerar la recuperación de datos.

`DataLoader` es un iterador que abstrae esta complejidad para nosotros en una API fácil.

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_dat, batch_size=64, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

#### Iterar a través del DataLoader
Hemos cargado ese conjunto de datos en `DataLoader` y podemos iterar a través del dataset según sea necesario. Cada iteración a continuación devuelve un lote de `train_features` y `train_labels` (que contienen `batch_size=64` para las características y etiquetas respectivamente). Debido a que especificamos `shuffle=True`, después de iterar sobre todos los lotes, los datos se mezclan (para un control más detallado sobre el orden de carga de datos, eche un vistazo a Samplers).

```python
# Display image and label
train_features, train_labels = next(iter(train_dataloader))
pritn(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

>[!tip] A Recordar
>`Datasets` Es una clase que todo datasets que puedas crear o que venga precargado hereda de ella y básicamente tiene allí guardado los objetos, si vos le pasas un índice te devuelve el elemento y no mucho más que eso.
>`DataLoader`: Se usan para obtener **_lotes_** de esos datasets y usarlos para entrenar
>

### Los fundamentos de Autograd
La función `Autograd` de PyTorch es parte de lo que hace que PyTorch sea flexible y rápido para crear proyectos de aprendizaje automático. Permite el cálculo rápido y sencillo de múltiples derivadas parciales (también conocidas como _gradientes_) en un cálculo complejo. Esta operación es fundamental para el aprendizaje de redes neuronales basado en backpropagation.

El poder de autograd proviene del hecho de que rastrea su computación dinámicamente en `tiempo de ejecución`, lo que significa que si su modelo tiene ramas de decisión o bucles cuyas longitudes no se conocen hasta el tiempo de ejecución, la computación aún se rastreará correctamente y obtendrá gradientes correctos para impulsar el aprendizaje, Esto, combinado con el hecho de que sus modelos están construidos en Python, ofrece mucha más flexibilidad que los frameworks que se basan en el análisis estático de un modelo de estructura más rígida para calcular gradientes.

#### ¿Para qué necesitamos Autograd?
Un modelo de aprendizaje automático es una *función*, con entradas y salidas. Para esta discusión, trataremos a las entradas como un vector de dimensión *i* $\vec{x}$, con elementos $x_{i}$. Entonces podemos expresar el modelo, *M*, como una función vectorial de la entrada: $\vec{y} = \vec{M}(\vec{x})$. (Tratamos el valor de la salida de M como un vector porque, en general, un modelo puede tener cualquier número de salidas).

Dado que principalmente hablaremos de autograd en el contexto del entrenamiento, nuestro resultado de interés será la pérdida del modelo. La *función de pérdida* L($\vec{y}$) = L($\vec{M}$($\vec{x}$)) es una función escalar de valor único que depende de la salida del modelo. Esta función expresa qué tan lejos estaba la predicción de nuestro modelo de la salida *ideal* de una entrada en particular. *Nota: después de este punto, a menudo omitiremos el signo del vector donde debería ser contextualmente claro, por ejemplo,* $y$ en lugar de $\vec y$.

Al entrenar un modelo, queremos minimizar la pérdida. En el caso idealizado de un modelo perfecto, eso significa ajustar sus pesos de aprendizaje, es decir, los parámetros ajustables de la función, de modo que la pérdida sea cero para todas las entradas. En el mundo real, significa un proceso iterativo de empujar los pesos de aprendizaje hasta que veamos que obtenemos una pérdida tolerable para una amplia variedad de entradas.

¿Cómo decidimos hasta dónde y en qué dirección empujar los pesos? Queremos *minimizar* la pérdida, lo que significa hacer que su primera derivada con respecto a la entrada sea igual a 0:
$\frac{\partial L}{\partial x} = 0$.

Sin embargo, recuerde que la pérdida no se deriva *directamente* de la entrada, sino que es una función de la salida del modelo (que es una función directa de la entrada), $\frac{\partial L}{\partial x}$ =
$\frac{\partial {L({\vec y})}}{\partial x}$. 

Por la regla de la cadena del cálculo diferencial, tenemos

$$\frac{\partial {L({\vec y})}}{\partial x} =
\frac{\partial L}{\partial y}\frac{\partial y}{\partial x} =
\frac{\partial L}{\partial y}\frac{\partial M(x)}{\partial x}$$

$\frac{\partial M(x)}{\partial x}$ es donde las cosas se ponen complejas.
Las derivadas parciales de las salidas del modelo con respecto a sus entradas, si tuviéramos que expandir la expresión usando la regla de la cadena nuevamente, involucrarían muchas derivadas parciales locales sobre cada peso de aprendizaje multiplicado, cada función de activación y cualquier otra transformación matemática en el modelo. La expresión completa para cada derivada parcial es la suma de los productos del gradiente local de *todas las rutas posibles* a través del grafo computacional que termina con la variable cuyo gradiente estamos tratando de medir.

En particular, los gradientes sobre los pesos de aprendizaje nos interesan: nos dicen *en qué dirección cambiar cada peso* para acercar la función de pérdida a cero. Dado que el número de tales derivadas locales (cada uno correspondiente a una ruta separada a través del grafo commputacional del modelo) tenderá a aumentar exponencialmente con la profundidad de una red neuronal, también lo hace la complejidad de calcularlas. 

![Imgur](https://i.imgur.com/vnYfIrR.png)


Aquí es donde entra en juego autograd: realiza un seguimiento del historial de cada cálculo. Cada tensor calculado en su modelo PyTorch lleva un historial de sus tensores de entrada y la función utilizada para crearlo. Combinado con el hecho de que las funciones de PyTorch destinadas a actuar sobre tensores tienen una implementación integrada para calcular sus propias derivadas, esto acelera enormemente el cálculo de las derivadas locales necesarias para el aprendizaje.

#### Un ejemplo sencillo
Eso fue mucha teoría, pero ¿cómo se ve usar autograd en práctica?

Comencemos con un ejemplo sencillo. Primero, haremos algunas importaciones, para graficar nuestros resultados:
```python
# %matplotlib inline

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
```

A continuación, creamos un tensor de entrada lleno de valores espaciados uniformemente en el intervalo $[0, 2\pi ]$  y especificaremos `requires_grad=True`. (Como la mayoría de las funciones que crean tensores, `torch.linespace()` acepta una opción `requires_grad` opcional).

Establecer este indicador significa que en cada cálculo que sigue, autograd acumulará el historial del cálculo en los tensores de salida de ese cálculo.

```python
a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
print(a)
```

A continuación realizaremos un cálculo y graficaremos su salida en términos de sus entradas:

```python
b = torch.sin(a)
plt.plot(a.detach(), b.detach())
```

Echemos un vistazo más de cerca al tensor `b`. Cuando lo imprimimos, veremos un indicador de que está rastreando su historial de cómputo:
```python
print(b)
```

Este `grad_fn` nos da una pista de que cuando ejecutemos el paso de backpropagation y calculemos los gradientes, necesitaremos calcular la derivada de `sin(x)` para todas las entradas de este tensor.

Realicemos algunos cálculos más:
```python
c = 2 * b
print(c)

d = c + 1
print(d)
```

Finalmente, calculemos una salida de un solo elemento. Cuando llamas a `.backward()` en un tensor sin argumentos, se espera que el tensor de llamada contenga solo un elemento, como es el caso cuando se calcula una función de pérdida.

```python
out = d.sum()
print(out)
```

Cada `grad_fn` almacenado con nuestros tensores le permite recorrer el cálculo hasta sus entradas con su atributo `next_functions`. Podemos ver a continuación que profundizar en este atributo en `d` nos muestra las funciones de gradiente para todos los tensores anteriores. Tenga en cuenta que `a.grad_fn` se informa como `None`, lo que indica que esta fue una entrada a la función sin historial propio.

```python
print('d:')

print(d.grad_fn)

print(d.grad_fn.next_functions)

print(d.grad_fn.next_functions[0][0].next_functions)

print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)

print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)

print('\nc:')

print(c.grad_fn)

print('\nb:')

print(b.grad_fn)

print('\na:')

print(a.grad_fn)
```


Con toda esta maquinaria en su lugar, ¿cómo sacamos las derivadas? Llama al método `backward()` en la salida y verifica el atributo `grad` de la entrada para inspeccionar los gradientes:
```python
out.backward()
print(a.grad)
plt.plot(a.detach(), a.grad.detach()) # detach es importante!!
```

>[!important]
>`backward()`: Sólo guarda los gradientes, será el optimizador dependiendo de la estrategia el que actualice los parámetros del modelo


#### Activar y desactivar Autograd
Hay situaciones en las que necesitará un control detallado sobre si autograd está habilitado. Hay varias formas de hacer esto, dependiendo de la situación.

El más simple es cambiar el indicador `requires_grad` en un tensor directamente:
```python
a = torch.ones(2, 3, requires_grad=True)
print(a)

b1 = 2 * a
print(b1)

a.requires_grad = False
b2 = 2 * a
print(b2)
```

En la celda de arriba, vemos que `b1` tiene un `grad_fn` (es decir, un historial de cálculo rastreado), que es lo que esperamos, ya que se derivó de un tensor, `a`, que tenía activado el autograd. Cuando desactivamos autograd explícitamente con `a.requires_grad=False`, ya no se rastrea el historial de cálculo, como vemos cuando calculamos `b2`.

Si solo necesita que Autograd se apague temporalmente, una mejor manera es usar `torch.no_grad():`
```python
a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = a + b
print(c1)

with torch.no_grad(): # deprecated -> torch.inference_mode():
    c2 = a + b

print(c2)

c3 = a * b
print(c3)
```

`torch.no_grad()` también se puede usar como decorador de funciones o métodos:

```python
def add_tensors1(x, y):
	return x + y

@torch.no_grad()
def add_tensors2(x, y):
	return x + y

a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = add_tensors1(a, b)
print(c1)

c2 = add_tensors2(a, b)
print(c2)
```

Hay un administrador de contexto correspondiente, `torch.enable_grad()`, para activar graduación automática cuando aún no lo está.

También se puede utilizar como decorador.

Finalmente, puede tener un tensor que requiera seguimiento de gradiente, pero desea una copia que no lo requiera. Para eso tenemos el método `detach()` del objeto `Tensor` crea una copia del tensor que se _separa_ del historial de cálculo:

```python
x = torch.rand(5, requires_grad=True)
y = x.detach()

print(x)
print(y)
```

Hicimos esto arriba cuando queríamos graficar algunos de nuestros tensores. Esto se debe a que matplotlib espera una matriz NumPy como entrada, y la conversión implícita de un tensor PyTorch a una matriz NumPy no está habilitada para tensores con require_grad=True. Hacer una copia separada nos permite avanzar.

#### Autograd y las Operaciones in situ
En todos los ejemplos de este cuaderno hasta ahora, hemos usado variables para capturar los valores intermedios de un cálculo. Autograd necesita estos valores intermedios para realizar cálculos de gradiente. **_Por esta razón, debe tener cuidado al usar operaciones in situ cuando use autograd._** Si lo hace, puede destruir la información que necesita para calcular las derivadas en la llamada `backward()`. PyTorch incluso lo detendrá si intenta una operación in situ en la variable de hoja que requiere autograduación, como se muestra a continuación.

```python
a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
torch.sin_(a)
```
