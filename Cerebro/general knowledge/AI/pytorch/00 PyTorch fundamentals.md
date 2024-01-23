---
sticker: emoji//1f6a7
type: knowledge
tags:
  - ai
  - pytorch
---
<iframe width="560" height="315" src="https://www.youtube.com/embed/Z_ikDlimN6A?si=FyN9GUMcHYR5Zuim" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
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

continuar aqui! -> https://youtu.be/KKbuqZJJRYU?si=hoBBpvRC1OFchflK&t=1718

### Datasets & DataLoaders

### Los fundamentos de Autograd
