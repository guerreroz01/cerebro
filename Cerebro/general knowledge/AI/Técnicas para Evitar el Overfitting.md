# Regularización de los pesos
___
Ahora que hemos caracterizado el problema del sobreajuste, podemos introducir algunas técnicas estándar para regularizar modelos. Recuerde que siempre podemos mitigar el sobreajuste saliendo y recopilando más datos de entrenamiento. Eso puede ser costoso, llevar mucho tiempo o estar completamente fuera de nuestro control, haciéndolo imposible a corto plazo. Por ahora, podemos asumir que ya tenemos tantos datos de alta calidad como nuestros recursos lo permitan y centrarnos en las técnicas de regularización.

Recuerde que en nuestro ejemplo de regresión polinomial podríamos limitar la capacidad de nuestro modelo simplemente modificando el grado del polinomio ajustado. De hecho, limitar el número de características es una técnica popular para mitigar el sobreajuste. Sin embargo, simplemente dejar de lado las características puede ser una medida demasiado drástica. Siguiendo con el ejemplo de regresión polinomial, considere lo que podría suceder con entradas de alta dimensión. Las extensiones naturales de polinomios a datos multivariados se denominan monomios, que son simplemente productos de potencias de variables. El grado de un monomio es la suma de las potencias. Por ejemplo, $x^{2}_1 x^{2}$ y $x_3 x^{2}_5$ son ambos monomios de grado 3.

Tenga en cuenta que el número de términos con grado $d$ aumenta rápidamente a medida que $d$ crece. Dadas las variables $k$, el número de monomios de grado $d$ es: $(\frac{k -1 + d}{k - 1})$ (es decir $k$ multiselección $d$.) Incluso pequeños cambios en el grado, digamos de $2$ a $3$ , aumentan drásticamente la complejidad de nuestro modelo. Por lo tanto, a menudo necesitamos una herramienta más fina para ajustar la complejidad de la función.

## Normas
___
Algunos de los operadores más útiles en álgebra lineal son _normas_. informalmente, la norma de un vector nos dice cuán _grande_ es un vector. La noción de _tamaño_ que se considera aquí no se refiere a la dimensionalidad sino a la magnitud de los componentes.

Puede notar que las normas se parecen mucho a las medidas de distancia. De hecho, la distancia euclidiana es una norma: en concreto es la norma $L_2$. Supongamos que los elementos en el vector $n-dimensional$ $x$ son $x_1, . . . , x_n$.
**La norma $L_2$ de $x$ es la raíz cuadrada de la suma de los cuadrados de los elementos del vector:**

**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$**
donde el subíndice $2$ a menudo se omite en las normas $L_2$, es decir, $\|\mathbf{x}\|$ es equivalente a $\|\mathbf{x}\|_2$. En código,

podemos calcular la norma $L_2$ de un vector de la siguiente manera.

```python
import torch
u = torch.tensor([3.0, -4.0])
torch.nom(u)
```

En el aprendizaje profundo, trabajamos más a menudo con la norma $L_2$ al cuadrado.

También encontrará con frecuencia **la noma $L_1$.** que se expresa como la suma de los valores absolutos de los elementos del vector:
**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**
En comparación con la norma $L_2$, está menos influenciada por valores atípicos. Para calcular la norma $L_1$, componemos la función de valor absoluto con una suma sobre los elementos.

```python
torch.abs(u).sum()
```

Tanto la norma $L_2$ como la norma $L_1$ son casos especiales de la norma $L_p$ más general:
$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

De manera análoga a la norma $L_2$ de los vectores, **la norma de Frobenius** de una matriz $\mathbf{x} \in \mathbb{R}^{m \times n}$ es la raíz cuadrada de la suma de los cuadrados de los elementos de la matriz:
**$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**
La norma de Frobenius satisface todas las propiedades de las normas vectoriales. Se comporta como si fuera una norma $L_2$ de un vector en forma de matriz. Invocar la siguiente función calculará la norma de Frobenius de una matriz.

```python
torch.norm(torch.ones(4, 9))
```


## Regularización $L_2$ (Weight Decay)
___
La regularización $L_2$ podría ser la técnica más utilizada para regularizar modelos de aprendizaje automático paramétrico. La técnica está motivada por la intuición básica de que entre todas las funciones $f$, la función $f = 0$ (que asigna el valor $0$ a todas las entradas) es, en cierto sentido, la más simple, y que podemos medir la complejidad de una función por su distancia a cero. Pero, ¿con qué precisión debemos medir la distancia entre una función y cero? No hay una sola respuesta  correcta. De hecho, ramas enteras de las matemáticas, incluidas partes del análisis funcional y la teoría de los espacios de Banach, están dedicadas a responder a este problema.

Una interpretación simple podría ser medir la complejidad de una función lineal $f(x) = w^{t} x$ por alguna norma de su vector de peso, por ejemplo, $||w||^{2}$. El método más común para asegurar un vector de peso pequeño es agregar su norma como término de penalización al problema de minimizar la pérdida.  Por lo tanto, reemplazamos nuestro objetivo original, _minimizar la pérdida de predicción en las etiquetas de entrenamiento_, con un nuevo objetivo, _minimizar la suma de la pérdida de predicción y el término de penalización_. Ahora, si nuestro vector de peso crece demasiado, nuestro algoritmo de aprendizaje podría enfocarse en minimizar la norma de peso ∥w∥2 vs. minimizar el error de entrenamiento. Eso es exactamente lo que queremos. Para ilustrar las cosas en el código, revivamos nuestro ejemplo anterior de la clase de regresión lineal. Allí, nuestra pérdida fue dada por
$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$
ecuerde que $\mathbf{x}^{(i)}$ son las características, $y^{(i)}$ son etiquetas para todos los ejemplos de datos $i$ y $(\mathbf{w}, b)$ son los parámetros de peso y sesgo, respectivamente. Para penalizar el tamaño del vector de peso, debemos agregar de alguna manera $\| \mathbf{w} \|^2$ a la función de pérdida, pero ¿cómo debería el modelo compensar la pérdida estándar por esta nueva penalización aditiva? En la práctica, caracterizamos esta compensación a través de la *constante de regularización* $\lambda$, un hiperparámetro no negativo que ajustamos usando datos de validación:

  

![Imgur](https://i.imgur.com/MJc7iXc.png)

Para $\lambda = 0$, recuperamos nuestra función de pérdida original. Para $\lambda > 0$, restringimos el tamaño de $\| \mathbf{w} \|$. Dividimos por $2$ por convención: cuando tomamos la derivada de una función cuadrática, $2$ y $1/2$ se cancelan, asegurando que la expresión para la actualización se vea bien y simple. El lector astuto podría preguntarse por qué trabajamos con la norma al cuadrado y no con la norma estándar (es decir, la distancia euclidiana). Hacemos esto por conveniencia computacional. Al elevar al cuadrado la norma $L_2$, eliminamos la raíz cuadrada, dejando la suma de cuadrados de cada componente del vector de peso. Esto hace que la derivada de la penalización sea fácil de calcular: la suma de las derivadas es igual a la derivada de la suma.

El efecto logrado es el que se ve en el siguiente gráfico donde tenemos un ejemplo con 2 pesos w1 y w2 como ejes y un función de pérdida graficada como sus curvas de nivel. El círculo celeste representa la restricción establecida a los pesos por la regularización l2 y el circulito amarillo es la combinación de pesos elegida por el modelo. Podemos observar que el modelo se acerca lo más posible al mínimo de la función mientras sigue respetando la restricción de tamaño.

![Imgur](https://i.imgur.com/L2T6rzd.gif)

Además, podría preguntarse por qué trabajamos con la norma $L_2$ en primer lugar y no, digamos, con la norma $L_1$. Una razón para trabajar con la norma $L_2$ es que impone una penalización descomunal a los pesos grandes del vector. Esto sesga nuestro algoritmo de aprendizaje hacia modelos que distribuyen el peso de manera uniforme entre una mayor cantidad de features. En la práctica, esto podría hacerlos más robustos al error de medición en una sola variable. Por el contrario, las penalizaciones de $L_1$ conducen a modelos que concentran los pesos en un pequeño conjunto de características disminuyendo los otros pesos a cero. Esto se llama *selección de características*, lo que puede ser deseable por otras razones.

## Regresión lineal de alta dimensión
___
Podemos ilustrar los beneficios de la pérdida de peso a través de un ejemplo sintético simple.

```python
%matplotlib inline
from torch import nn
from matplotlib_inline import backend_inline
```

Primero, **generamos algunos datos como antes**
**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**

Elegimos que nuestra etiqueta sea una función lineal de nuestras entradas, alterada por el ruido gaussiano con media cero y desviación estándar de 0,01. Para que los efectos del sobreajuste sean pronunciados, podemos aumentar la dimensionalidad de nuestro problema a $d = 200$ y trabajar con un pequeño conjunto de entrenamiento que contiene solo 20 ejemplos. Reutilizaremos varias de las funciones que usamos en la clase 2 de regresión lineal.

```python
import torch.utils.data as data

def synthetic_data(w, b, num_examples):
	X = torch.normal(0, 1, (num_examples, len(w)))
	y = torch.matmul(X, w) + b
	y += torch.normal(0, 0.01, y.shape)
	return X, y.reshape((-1, 1))

def load_array(data_arrays, batch_size, is_train=True):
	dataset = data.TensorDataset(*data_arrays)
	return data.DataLoader(dataset, batch_size, shuffle=is_train)


n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = synthetic_data(true_w, true_b, n_train)
train_iter = load_array(train_data, batch_size)
test_data = synthetic_data(true_w, true_b, n_test)
test_iter = load_array(test_data, batch_size, is_train=False)
```

### Implementación desde cero
___
A continuación, implementaremos la regularización de los pesos desde cero, simplemente agregando la penalización de $L_2$ al cuadrado a la función objetivo original.

### Definiendo el modelo
___
Primero, definiremos un modelo de regresión lineal e inicializaremos aleatoriamente sus parámetros. Nótese que también agregamos un método loss que calcula el error cuadrático medio.

```python
class LinearRegression(nn.Module):
	def __init__(self, num_inputs, sigma=0.01):
		super().__init__()
		self.w = nn.Parameter(torch.normal(0, sigma, (num_inputs, 1)))
		self.b = nn.Parameter(torch.zeros(1, requires_grad=True))

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		return torch.matmul(X, self.w) + self.b

	def loss(self, y_hat, y):
		l = (y_hat -y.reshape(y_hat.shape)) ** 2 / 2
		return l.mean()
```

### **Definiendo la Norma de Penalización $L_2$**
___
Quizás la forma más conveniente de implementar esta penalización es elevar al cuadrado todos los términos y sumarlos.

```python
def l2_penalty(w):
	return (w ** 2).sum() / 2
```

continuar aqui! https://youtu.be/SqoZKP5yisM?si=3VKDkTu8N7RvM_0I&t=1282