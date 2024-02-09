# Redes Neuronales Recurrentes (RNN)
_______

## Modelo de Secuencias:
___
Hasta ahora hemos trabajado con series de datos donde a cada entrada le corresponde una salida. Por ejemplo, a una imagen le corresponde una categoría. A una serie de indicadores biométricos le corresponde un diagnóstico médico.

En procesamiento de lenguajes naturales, nuestras salidas y nuestras entradas tienen una característica distinta.

Veamos un ejemplo:
> Usted tiene 16 años. Está prohibido vender alcohol a menores de 18 años. No puedo venderle esa botella

En el ejemplo anterior, tenemos tres afirmaciones, donde la última es una conclusión de las dos anteriores. En este sentido, cuando trabajamos con lenguajes tenemos el problema de lo próximo que se dice, depende de lo que se dijo antes. Es decir, estamos trabajando con secuencias temporales.

Peor aún, muchas veces la última salida, debe ser tratada como una nueva entrada. Piensa en el ejemplo anterior, si usted vive en un país latinoamericano o europeo, al llegar a **menores de** intuía que la edad límite sería **18 años**. Eso es porque como ciudadano de su país, sabe que esa es la ley. Es decir **18 años** podría haber sido una predicción, una salida de su red. Además, al haber predicho **18 años** ahora podemos concluir que **No puedo venderle esa botella**. Si la ley dijera que los menores de **14 años** pueden comprar alcohol, la segunda frase carecería de sentido. Es decir **18 años** es una predicción, una salida en un momento, que luego se convierte en una entrada o un _feature_ en otro.

<span style="color:#ff0000">
Es por lo anterior que se dice que estos modelos son modelos autoregresivos: las salidas luego se convierten en entradas, como en un problema recursivo.</span>

La naturaleza autoregresiva de nuestro modelo hace que debamos considerar la calidad de nuestras predicciones. Volviendo al ejemplo anterior, si nuestra predicción hubiera sido **14 años** en lugar de **18 años**, la conclusión final de nuestra frase sería distinta a la que hemos obtenido. Pequeños errores en nuestras predicciones pueden acumularse a lo largo del tiempo y generar resultados absurdos.

## Modelos markovianos y variables ocultas
___
Volvamos de nuevo a nuestro ejemplo de oraciones.
> Usted tiene 16 años. Está prohibido vender alcohol a menores de 18 años. No puedo venderle esa botella.

Supongamos de nuevo que queremos predecir **18 años**. La cantidad de escritas hasta ese momento es 11. Luego de predecir **18 años**, la cantidad de palabras aumetó a 13. Es decir, conforme predecimos y agregamos información, nuestro modelo debe responder a la cantidad creciente de palabras o ejemplos.

Recordemos que todas estas herramientas nacieron de la estadística, por lo que nuestras predicciones se basaran en considerar la probabilidad de que diferentes palabras ocurran en simultaneo. Esto es verdaderamente un problema: mientras más palabras tenemos, menos probable es que vuelvan a ocurrir. Si ocurren infrecuentemente, necesitaremos aumentar cada vez más la cantidad de ejemplos de nuestros datos. Esto puede ser un problema incluso para oraciones cortas. Una alternativa para paliar este problema es limitar la cantidad de palabras que miraremos hacia atrás.

<span style="color:#ffff00">Otra alternativa a esto es trabajar con variables ocultas</span>. Las variables ocultas son cantidades que de alguna manera agrupan la información de todos los casos anteriores. Por ejemplo:

>Usted es menor de edad. No puedo venderle esa botella.

Hemos resumido toda la información de dos oraciones en una sola mucho más corta.

De la misma manera que buscamos representaciones abstractas para palabras por medio de _tokens_, usaremos esos tokens para generar nuevas variables que resuman la información anterior. Es decir, generaremos una variable que de alguna manera tiene toda la información de **Usted es menor de edad**.

Al trabajar con variables ocultas, esperamos reemplazar todas las palabras anteriores con el último valor de la variable oculta. Así, nuestro problema que antes veía 13 variables o tokens ahora ve uno solo. Esto nos permite simplificar nuestro modelo para trabajar con **modelos makovianos de primer orden**.

Sin entrar en mucho detalle, los modelos markovianos son el tipo de modelos autoregresivos como los que hemos descrito hasta ahora.

## Esbozo de la noción de modelo de Markov
___
- Tenemos un estado (las últimas palabras escritas) a la cual llegamos a partir de un estado inicial bien definido.
- Tenemos una historia de estados pasados que afecta a estados futuros (cada palabra predicha o dentro de nuestro dataset)
- Hay una probabilidad asociada a cada cambio de estado.
- Queremos predecir cual será el próximo estado de un grupo finito de estados (la próxima palabra).
Al trabajar con un modelo markoviano sobre variables ocultas, esperamos que variable oculta resuma con tanta fidelidad los tokens pasados que solo necesitemos la variable oculta más reciente. Al necesitar solo la más reciente, se dice que es un modelo markoviano de primer orden (requiera solo una variable anterior). La razón por la que buscamos trabajar con modelos de primer orden es que son menos costosos computacionalmente.

En resumen nuestra propuesta para generar modelos de lenguaje consistirá en lo siguiente:
1. Tomaremos texto para crear nuestro dataset
2. Transformaremos nuestro texto en algún tipo de representación simbólica (_tokens_).
3. De esta manera, nuestro modelo de lenguaje se convertirá en un problema de clasificación: Dadas las palabras anteriores ¿Cuál es la siguiente palabra?
	1. Decimos que es un problema de clasificación, porque cada una de nuestras palabras es una categoría.
4. Para crear nuestro modelo de lenguaje, usaremos variables ocultas en el contexto de un modelo markoviano.
	1. La justificación para esto es que el lenguaje tiene características de un modelo markoviano.

## Redes neuronales recurrentes
___
En la sección anterior intentaremos argumentar que el lenguaje puede modelarse con un modelo markoviano. Además, propusimos trabajar con modelos markovianos con variables ocultas. La idea de trabajar con variables ocultas era poder trabajar con un modelo markoviano de primer orden. Queremos usar estos modelos de primer orden porque sabemos que nos permitirán ahorrar uso de memoria, así como disminuir el uso de  recursos computacionales.

Nuestra propuesta para trabajar con variables ocultas, será trabajar con la unidades ocultas de un [[Perceptron Multicapa]].
![[Pasted image 20240125230147.png]]

$$\mathbf{O} = \mathbf{H} \mathbf{W} + \mathbf{b}$$
$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W} + \mathbf{b})$$

Sin embargo, como trabajaremos con secuencias temporales, nuestra entrada al tiempo $t$, debe depender del tiempo $t - 1$. Es decir, la próxima palabra debe depender de las palabras anteriores. En un perceptrón multicapa, esa dependencia temporal no está presente. Es por esto que debemos reestructurar nuestra capa para que permita generar modelos autoregresivos de secuencias. Dado queremos usar las variables ocultas como cantidades que resumen toda la información anterior, son estas cantidades las que tendrán  una dependencia temporal.

$$\mathbf{O}_{t} = \mathbf{H}_{t} \mathbf{W}_{O} + \mathbf{b}$$
$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{X} + \mathbf{H}_{t-1} \mathbf{W}_{H} + \mathbf{b}_h).$$

Notemos que $\mathbf{H}_t$ depende del valor anterior, $\mathbf{H}_{t-1}$ y del nuevo valor $\mathbf{X}_t$. Esta dependencia temporal es la que hace que nuestra nueva red neuronal sea una *red neuronal recurrente*. Recordemos si nuestro modelo está correctamente entrenado la salida $\mathbf{O}_t$ debe coincidir con el resultado correcto o *grounding truth* de $\mathbf{X}_{t+1}$. Esta era la naturaleza autoregresiva de nuestros modelos.

En la siguiente figura mostramos el proceso de cálculo nuestra capa recurrente

![An RNN with a hidden state.](http://d2l.ai/_images/rnn.svg)

1. Tenemos una capa densa con función de activación $\phi$ que toma nuestra matriz de diseño $\mathbf{X}_t$ y nuestra variable oculta $\mathbf{H}_{t-1}$.
2. A la salida generamos nuestra nueva variable oculta $\mathbf{H}_t$.
3. Con $\mathbf{H}_t$ y otra capa densa generamos nuestra salida $\mathbf{O}_t$

Muchas veces, en el paso 1 lo que se hace es concatenar $\mathbf{X}_t$ y $\mathbf{H}_{t-1}$ para de esa manera definir una única matriz de pesos. A continuación mostramos como esta concatenación genera el mismo resultado.

```python
import torch

X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)
torch.matmul(X, W_xh) + torch.matmul(H, W_hh)

torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
```

## Problemas con nuestras predicciones
___
Un último punto que hemos evitado hasta ahora es el problema de la predicción en secuencias temporales. Dijimos más arriba que la salida $\mathbf{O}_t$ debe coincidir con el resultado correcto o *grounding truth* de $\mathbf{X}_{t+1}$. Sin embargo, para entrenar correctamente nuestro modelo debemos cada tanto usar la salida $\mathbf{O}_t$ en lugar de $\mathbf{X}_{t+1}$. Si usamos siempre nuestra *grounding truth* en lugar de nuestra predicción, corremos el riesgo de que nuestra red no aprenda a adaptarse a malas predicciones que genera. Al mismo tiempo, nuestra red aprende a modelar las secuencia de entrenamiento, pero puede no saber que hacer con oraciones nuevas.

El ejemplo sería como el siguiente:

Entrenamos una red con "El ingenioso hidalgo Don Quijote de la Mancha". Luego usamos la red para completar:

> *En un lugar de la Mancha de cuyo nombre prefiero no ...*

La red predice "recordar", en lugar de "acordarme". En el siguiente paso, si usamos "acordarme" la nueva predicción podría ser "vivía", pero si usaremos "recordar" la predicción podría ser "Residía". Es decir, al usar solo la *grounding truth* en lugar de nuestra predicción, el modelo no aprende a corregir sus errores, y se queda "estancado" en los datos de entrenamiento. El mayor problema es que cuando tengamos un modelo funcionando, NO HAY *GROUNDING TRUTH*. Esto hace que nuestro modelo pueda fallar estrepitosamente si no usamos nuestra predicción como entrada al modelo. Pero al mismo tiempo, si usamos siempre nuestra predicción, los errores se acumular paulatinamente.

Más adelante hablaremos de la técnica llamada **teacher forcing** y como elegir cuando usar la predicción y el *grounding truth* para generar redes que puedan adaptarse a la variabilidad de nuestra predicción.

## Preliminares a la implementación de RNN
___
Antes de discutir pasar a implementar una RNN desde cero, queremos discutir algunos temas más que serán importantes conocer.

### Muestro de secuencias.
___
Cuando tenemos que elegir que ejemplos debemos usar de un dataset que no contiene secuencias, simplemente mezclabamos aleatoriamente los ejemplos y luego los usábamos para entrenar nuestras redes.

Sin embargo, en secuencias temporales no podemos hacer esto. Si mezclamos aleatoriamente podemos terminar generando secuencias sin sentido.

>En un lugar de la Mancha de cuyo nombre prefiero no acordarme

luego de mezclarlo

>nombre En de de no lugar prefiero la Mancha cuyo acordarme un

Por esta razón debemos generar particiones y mezclarlas. Por ejemplo podemos generar particiones de 4 elementos.

> [En un lugar de]  [la Mancha de cuyo]  [nombre prefiero no acordarme]  

> [la Mancha de cuyo]  [nombre prefiero no acordarme]  [en un lugar de]

Además de, podemos elegir un offset o desplazamiento. En el ejemplo anterior, un offset de 1 generaría:

> En [un lugar de la]  [Mancha de cuyo nombre]  [prefiero no acordarme, no]


## Perplejidad
___

La perplejidad es una métrica que es usada en procesamiento de lenguajes naturales para tener una idea de que tan "convencido" está  nuestro modelo de la siguiente palabra que adivinará. Como métrica está relacionada a la entropía y la entropía cruzada, por lo que volveremos a usar los ejemplos del juego que discutimos anteriormente.

Recordemos nuestro juego:
* Materiales:
* una bolsa o recipiente opaco
* 4 pelotas con los números 1, 2, 3, 4
* Preparativos:
* Se colocan las pelotas en la bolsa
* El primer jugador saca una de las pelotas de la bolsa
* Objetivo general: Adivinar con el menor número de preguntas posibles cuál el número de la pelota que tiene el primer jugador .
* Solo pueden hacerse preguntas que tengan como respuestas sí o no.

Recordemos la estrategia optima:
```
1. Preguntar: "¿El número es par?"
A. Si la respuesta es sí, preguntar: "¿Es el número 4?"
a. Si la respuesta es sí, sabemos que es el número 4, hemos ganado.
b. Si la respuesta es no, sabemos que es el número 2, hemos ganado.
B. Si la respuesta es no, preguntar: "¿Es el número 3?"
a. Si la respuesta es sí, sabemos que es el número 3, hemos ganado.
b. Si la respuesta es no, sabemos que es el número 1, hemos ganado.
```

Recordemos la entropía de nuestra estrategia:  

|  | $1$ | $2$ | $3$ | $4$ | total |  |
| ---- | ---- | ---- | ---- | ---- | :--: | ---- |
| probabilidad de ocurrir | $\dfrac{1}{4}$ | $\dfrac{1}{4}$ | $\dfrac{1}{4}$ | $\dfrac{1}{4}$ | - |  |
| número de preguntas | $2$ | $2$ | $2$ | $2$ | - |  |
| producto | $\dfrac{1}{2}$ | $\dfrac{1}{2}$ | $\dfrac{1}{2}$ | $\dfrac{1}{2}$ | $2$ |  |

  
Ahora preguntamos, ¿cuantas opciones posibles tenemos en nuestro juego?

> Como todas las pelotas son equiprobables, tenemos 4 opciones distintas.

La pregunta ahora es que pasará en nuestros segundo juego cuando preguntemos ¿cuantas opciones posibles hay?

Segundo juego:
* Materiales:
* una bolsa o recipiente opaco
* 8 pelotas con los números 1, 1, 1, 1, 2, 2, 3, 4
* Preparativos:
* Se colocan las pelotas en la bolsa
* El primer jugador saca una de las pelotas de la bolsa
* Objetivo general: Adivinar con el menor número de preguntas posibles cuál el número de la pelota que tiene el primer jugador .
* Solo pueden hacerse preguntas que tengan como respuestas sí o no.
  
Estrategia optima
```
1. Preguntar: "¿Es el número 1?"
A. Si la respuesta es sí, hemos ganado."
B. Si la respuesta es no, preguntar: "¿Es el número 2?"
a. Si la respuesta es sí, hemos ganado.
b. Si la respuesta es no, preguntar: "¿Es el número 3?.
I. Si la respuesta es sí, sabemos que es el número 3, hemos ganado.
I. Si la respuesta es no, sabemos que es el número 4, hemos ganado.
```

| |$1$|$2$|$3$|$4$|total|
|---|---|---|---|---|:-:|
|probabilidad de ocurrir|$\dfrac{4}{8}$|$\dfrac{2}{8}$|$\dfrac{1}{8}$|$\dfrac{1}{8}$|-|
|número de preguntas|$1$|$2$|$3$|$3$|-|
|producto|$\dfrac{1}{2}$|$\dfrac{1}{2}$|$\dfrac{3}{8}$|$\dfrac{3}{8}$|$1.75$|

  

Dado que la probabilidad de la pelota 1 es mucho mayor que las demás, no tiene sentido decir que tenemos 4 opciones equiprobables. Por el contrario, debemos tener menos de 2 opciones. Esto es porque el 75% de las veces caeremos en la pelota con el número 1 o la pelota con el número 2. La pregunta es, como encontramos esta cantidad de opciones en promedio.