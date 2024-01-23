# ¿Qué es el Depp Learning?
Supongamos que queremos ir a tomar un café y alguien decide sacar su teléfono para pedir ayuda al asistente de Google. Si alguien dice "Hey Google", se despierta el sistema de reconocimiento de voz del teléfono. Si luego decimos "direcciones a la cafetería más cercana", el teléfono mostrará rápidamente la transcripción del comando y reconocerá que estamos pidiendo direcciones y lanzará la aplicación Mapas para cumplir con nuestro pedido. Una vez lanzada, la aplicación de Mapas identificará una serie de rutas con un tiempo de tránsito previsto para cada una.

Esta historia demuestra que el lapso de unos pocos segundos, nuestras interacciones cotidianas con un teléfono inteligente puede involucrar varios modelos de aprendizaje automático.

Imaginemos escribir un programa para responder a una palabra de activación como "Alexa", "Ok Google" u "Oye Siri". Intentar codificarlo en una habitación solo con nada más que una computadora y un editor de código es un problema difícil. Cada segundo, el micrófono recogerá aproximadamente 44000 muestras.  Cada muestra es una medida de la amplitud de la onda sonora. ¿Qué regla podría mapear de manera confiable desde un fragmento de audio sin procesar hasta mediciones confiables {si, no} sobre si el fragmento contiene la palabra de activación? Si estás atascado, no te preocupes. Tampoco sabemos cómo escribir un programa de este tipo desde cero. Es por eso que usamos el `aprendizaje automático`.
![[Pasted image 20240123130642.png]]

El truco es que, a menudo incluso cuando no sabemos cómo decirle a una computadora explícitamente cómo mapear de ciertas entradas a salidas, sí somos capaces de realizar la proeza cognitiva por nosotros mismos. En otras palabras, incluso si no sabemos cómo programar una computadora para que reconozca la palabra "Alexa", si eras capaz de reconocerla. 

Armados con esta habilidad, podemos recopilar un enorme `conjunto de datos` que contenga ejemplos de audio y etiquetar aquellos que contengan la palabra de activación y aquellos que no. En el enfoque de aprendizaje automático, no intentamos diseñar un sistema _explícitamente_ para reconocer palabras de activación. En su lugar, definimos un programa flexible cuyo comportamiento está determinado por una serie de **parámetros**. Luego usamos el conjunto de datos para determinar el mejor conjunto posible de parámetros, aquellos que mejoran el desempeño de nuestro programa con respecto a alguna medida de desempeño en la tarea de interés.

![[Pasted image 20240123131341.png]]

Todas las combinaciones de en éste caso 4 parámetros es lo que llamaríamos _Modelo_, Una vez que ya se ha determinado que nuestro modelo tendrá sólo 4 parámetros, todas las combinaciones posibles de esos 4 parámetros es lo que llamaremos `familia de modelos`.

Puedes pensar en los parámetros como perillas que podemos girar, manipulando el comportamiento del programa. Una vez fijados los parámetros, llamamos al programa `modelo`. El conjunto de todos los programas distintos (mapeos de entrada-salida) que podemos producir simplemente manipulando los parámetros se llama una `famila de modelos`. Y el metaprograma que usa nuestro conjunto de datos para elegir los parámetros se llama un `algoritmo de aprendizaje`.

Antes que podamos seguir adelante y activar el algoritmo de aprendizaje,  tenemos que definir el problema con precisión, determinar la naturaleza exacta de las entradas y salidas, y elegir una familia modelo apropiada. En este caso, nuestro modelo recibe un fragmento de audio  como entrada, y el modelo genera una selección entre {si, no} como salida. Si todo va de acuerdo al plan las conjeturas del modelo serán suelen ser correcto en cuanto a si el fragmento contiene la palabra de activación.

Si elegimos la familia de modelos adecuada, debe existir una configuración de las perillas tal que el modelo dispara "si", cada vez que escucha la palabra "Alexa". Debido a que la elección exacta de la palabra de activación es arbitraria, probablemente necesitaremos una familia modelo lo suficientemente rica como para que, a través de otra configuración de las perillas, podría disparar "si" sólo al escuchar la palabra "tomate". Esperamos que la misma familia de modelos sea adecuada para reconocer tanto a "Alexa" como a "tomate" porque parecen, intuitivamente, ser tareas similares.

Sin embargo, es posible que necesitemos una familia diferente de modelos completamente si queremos tratar con entradas o salidas fundamentalmente diferentes, digamos si quisiéramos mapear de imágenes a subtítulos, o de oraciones en inglés a oraciones en chino.

Como puede suponer, si configuramos todas las perillas al azar, es poco probable que nuestro modelo reconozca "Alexa", "tomate", o cualquier otra palabra. En el aprendizaje automático, el `aprendizaje` es el proceso por el cual descubrimos la configuración correcta de las perillas forzando el comportamiento deseado de nuestro modelo. En otras palabras, `entrenamos` nuestro modelo con datos.

Como se muestra en la siguiente figura, el proceso de entrenamiento generalmente se parece a lo siguiente:
1. Comenzar con un modelo con parámetros inicializado aleatoriamente que no puede hacer nada útil.
2. Tomar algunos de los datos (p. ej., fragmentos de audio y etiquetas {si, no} correspondientes).
3. Ajustar las perillas para que el modelo sea menos inútil con respecto a esos ejemplos.
4. Repita los pasos 2 y 3 hasta que el modelo quede increíble.

![[Pasted image 20240123133418.png]]

Para resumir, en lugar de codificar un reconocedor de palabras de activación, codificamos un programa que puede `aprender` a reconocer palabras de activación, si lo entrenamos con un gran conjunto de datos etiquetados. Puede pensar en este acto de determinar el comportamiento de un programa entrenándolo con un conjunto de datos como _programación con datos_. Es decir, podemos "programar" un detector de gatos proporcionando a nuestro sistema de aprendizaje automático con muchos ejemplos de gatos y perros. De esta forma, el detector eventualmente aprenderá a emitir un nuevo positivo muy grande si es un gato, un número negativo muy grande si es un perro, y algo más cercano a cero si no está seguro, y esto apenas araña la superficie de lo puede hacer el aprendizaje automático. El aprendizaje profundo o Deep Learning, que explicamos con más detalle más adelante, es sólo uno entre muchos métodos populares para resolver problemas de aprendizaje automático.

## Componentes Claves:
En nuestro ejemplo de palabras de activación, describimos un conjunto de datos que consta de fragmentos de audio y etiquetas binarias, y nos dio una idea de cómo podríamos entrenar un modelo para aproximar un mapeo de fragmentos de audio a clasificaciones. Este tipo de problema, donde tratamos de predecir una etiqueta desconocida designada basado en entradas conocidas dado un conjunto de datos que consta de ejemplos por las que se conocen las etiquetas, se llama <span style="color:#f519e3">_aprendizaje supervisado_.</span> Este es solo uno entre muchos tipos de problemas de aprendizaje automático. En primer lugar, nos gustaría arrojar más luz sobre algunos componentes básicos eso nos seguirá, sin importar qué tipo de problema de aprendizaje automático abordemos.
1. Los<span style="color:#ffff00"> _datos_ </span>de los que podemos aprender.
2. Un <span style="color:#ffff00">_modelo_ </span>de cómo transformar los datos.
3. Una<span style="color:#ffff00"> _función objetivo_ </span>que cuantifica qué tan bien (o mal) está funcionando el modelo.
4. Un <span style="color:#ffff00">_algoritmo de optimización_</span> para ajustar los parámetros del modelo tratando de optimizar la función de objetivo.

## Datos
No hace falta decir que no se puede hacer ciencia de datos sin datos. Podríamos perder cientos de páginas reflexionando sobre qué constituyen precisamente los datos, pero por ahora, vamos a enfocarnos en el lado práctico y en las propiedades clave de las que debe ocuparse. En general, nos ocupamos de una colección de ejemplos. Para trabajar con datos de manera útil, normalmente es necesario llegar a una `representación numérica` adecuada.

Cada `ejemplo` generalmente consta de un conjunto de atributos llamados `características` (o _features_), a partir de la cual el modelo debe hacer sus predicciones. En los problemas de aprendizaje supervisado anteriores, lo que hay que predecir es un atributo especial que se designa como la `etiqueta`.

![[Pasted image 20240123135658.png]]

Cuando cada ejemplo se caracteriza por el mismo número de valores numéricos, decimos que los datos consisten en vectores de `longitud fija` y describimos la longitud constante de los vectores como la `dimensionalidad` de los datos.

Una de las principales ventajas del aprendizaje profundo sobre los métodos tradicionales es la gracia comparativa con la que los modelos modernos puede manejar datos de `longitud variable`.

Generalmente, cuanto más datos tenemos, más fácil se vuelve nuestro trabajo. Cuando tenemos más datos, podemos entrenar modelos más potentes y confiar menos en suposiciones preconcebidas. El cambio de escala en la cantidad de datos disponible es un importante contribuyente al éxito del aprendizaje profundo moderno. Para recalcar el punto, muchos de los modelos más emocionantes en el aprendizaje profundo no funcionan sin grandes conjuntos de datos. Algunos otros trabajan en el régimen de datos pequeños, pero no son mejores que los enfoques tradicionales.

Finalmente, no es suficiente tener muchos datos y procesarlos inteligentemente. Necesitamos los datos correctos. Si los datos están llenos de errores, o si las características elegidas no son predictivas de la cantidad objetivo de interés, el aprendizaje va a fallar.

Además, el rendimiento predictivo deficiente no es la única consecuencia potencial. En aplicaciones sensibles de aprendizaje automático, como la vigilancia predictiva, la selección de currículums y los modelos de riesgo utilizados para préstamos, debemos estar especialmente atentos a las consecuencias de los datos basura. Un modelo de falla común ocurre en conjuntos de datos donde algún grupo de personas no están representados en los datos de entrenamiento. Imagine aplicar un sistema de reconocimiento de cáncer de piel en la naturaleza que nunca había visto piel negra antes. La falla también puede ocurrir cuando los datos no subrepresentan simplemente algunos grupos sino que reflejan prejuicios sociales. Por ejemplo, si las decisiones de contratación anteriores se utilizan para entrenar un modelo predictivo que se usará para filtrar [[currículums]], entonces los modelos de aprendizaje automático podrían inadvertidamente capturar y automatizar injusticias históricas. Tenga en cuenta que todo esto puede suceder sin el científico de datos conspirando activamente, o incluso siendo consciente.

## Modelos
La mayor parte del aprendizaje automático implica transformar los datos en algún sentido. Es posible que queramos construir un sistema que ingiera fotos y prediga la ubicación de las caritas sonrientes.

Por `modelo` , denotamos la maquinaria computacional para la ingesta de datos de un tipo, que escupe predicciones de un tipo posiblemente diferente. En particular, estamos interesados en modelos estadísticos que se puedan estimar a partir de los datos.

Si bien los modelos simples son perfectamente capaces de abordar problemas apropiadamente simples, los problemas en los que nos centramos en este libro amplían los límites de los métodos clásicos del aprendizaje automático tradicional. Estos modelos consisten en muchas transformaciones sucesivas de los datos que están encadenados de arriba a abajo, de ahí el nombre de `aprendizaje profundo`. De camino a discutir modelos profundos, también discutiremos algunos métodos más tradicionales.

## Funciones Objetivo
Anteriormente, presentamos el aprendizaje automático como aprendizaje a partir de la experiencia. Cuando decimos **aprender** aquí, nos referimos a mejorar en alguna tarea con la experiencia. Pero ¿quién puede decir qué constituye una mejora? No es ilógico imaginar que podríamos proponer actualizar nuestro modelo, y algunas personas podrían estar en desacuerdo sobre si la actualización propuesta constituye una mejora o declive.

Para desarrollar un sistema matemático formal de aprendizaje automático, necesitamos tener medidas formales de cuán buenos (o malos) son nuestros modelos. En el aprendizaje automático y la optimización en general, las llamamos _funciones objetivo_ . Por convención generalmente definimos funciones objetivo en las que mientras más bajo, mejor, y por eso a estas funciones a veces se las denomina **funciones de pérdida**. Esto es simplemente una convención.

Al tratar de predecir valores numéricos, la función de pérdida más común es el _error cuadrático medio_, es decir, el cuadrado de la diferencia entre la predicción y la realidad.

Para la clasificación, el objetivo más común es minimizar la tasa de error, es decir, la fracción de ejemplos en los que nuestras predicciones no están de acuerdo con la verdad fundamental.

Algunos objetivos (p. ej., error cuadrático) son fáciles de optimizar. Otros (por ejemplo, la tasa de error) son difíciles de optimizar directamente, debido a la no diferenciabilidad u otras complicaciones. En estos casos, es común optimizar un _objetivo sustituto_.

Tipicamente, la función de pérdida se define con respecto a los parámetros del modelo y depende del conjunto de datos. Aprendemos los mejores valores de los parámetros de nuestro modelo al minimizar la pérdida incurrida en un conjunto que consta de una serie de ejemplos recopilados para el entrenamiento. Sin embargo, hacerlo bien con los datos de entrenamiento no garantiza que lo haremos bien con los datos no vistos. Por lo tanto normalmente queremos dividir los datos disponibles en dos particiones: el `conjunto de datos de entrenamiento` y el `conjunto de datos de prueba` que se ofrece para su evaluación informando cómo se desempeña el modelo en ambos.

Se podría pensar en el rendimiento del entrenamiento como si fuera las puntuaciones de un estudiante en los exámenes de práctica solía prepararse para algún examen final real. Incluso si los resultados son alentadores, eso no garantiza el éxito en el examen final. En otras palabras el rendimiento de la prueba puede desviarse significativamente del rendimiento del entrenamiento. Cuando un modelo se desempeña bien en el conjunto de entrenamiento pero logra generalizar a datos no vistos, decimos que es **_sobreajuste_**. En términos de la vida real, esto es como reprobar el examen real, a pesar de tener buenos resultados en los exámenes de práctica.

## Algoritmos de Optimización
Una vez que tenemos alguna fuente de datos, un modelo, y una función objetivo bien definida, necesitamos un algoritmo capaz de obtener los mejores parámetros posibles para minimizar la función  de pérdida.

Los algoritmos de optimización populares para el aprendizaje profundo se basan en un enfoque llamado [[descenso de gradiente]]. En resumen, en cada paso, este método se fija, para cada parámetro, en qué dirección se movería la pérdida del conjunto de entrenamiento si se perturbara ese parámetro solo un pequeña cantidad. Luego se actualiza el parámetro en la dirección que puede reducir la pérdida.

![[Pasted image 20240123144520.png]]

## Características del Deep Learning
Aunque el aprendizaje profundo es un subconjunto del aprendizaje automático, el vertiginoso conjunto de algoritmos y aplicaciones dificulta evaluar cuáles podrían ser específicamente los ingredientes para el aprendizaje profundo. Eso es tan difícil como tratar de precisar los ingredientes necesarios para la pizza, ya que casi todos los componentes son sustituibles, pero se puede encontrar los siguientes aspectos.

### Entrenamiento de Extremo a Extremo
Como hemos descrito, el aprendizaje automático puede usar datos para aprender transformaciones entre entradas y salidas. Al hacerlo a menudo es necesario representar los datos de una manera adecuada para que los algoritmos transformen dichas representaciones en la salida.

El _Aprendizaje profundo es profundo_ precisamente en el sentido que sus modelos aprenden muchas capas de transformaciones, donde cada capa ofrece un nivel de representación. Por ejemplo, las capas cerca de la entrada pueden representar detalles de bajo nivel de los datos, mientras que las capas están más cerca de la salida de clasificación pueden representar conceptos más abstractos utilizados para la discriminación. 

Dado que el **_aprendizaje de características_** apunta a encontrar las características para encontrar la mejor representación de los datos, el aprendizaje profundo puede denominarse como un **aprendizaje de características multinivel**. Resulta que estos modelos de muchas capas son capaces de abordar datos de percepción de bajo nivel de una manera que las herramientas anteriores no podían. Podría decirse que el elemento común más significativo en los métodos de aprendizaje profundo es el uso de _**entrenamiento de extremo a extremo**_.

En el pasado, la parte crucial de aplicar el aprendizaje automático a estos problemas consistía en idear características diseñadas manualmente para transformar los datos en alguna forma para mejorar la performance de los algoritmos tradicionales ( a los que nos referiremos como modelos superficiales). Desafortunadamente, es muy poco lo que los humanos pueden lograr con ingenio en comparación con una evaluación consistente sobre millones de elecciones realizadas automáticamente por un algoritmo. Cuando el aprendizaje profundo se hizo cargo, estos ingenieros de características fueron reemplazados por filtros optimizados automáticamente lo que brinda una precisión superior. De este modo, una ventaja clave del aprendizaje profundo es que reemplaza no solo los modelos superficiales al final de los pipelines de aprendizaje tradicionales, sino también el proceso laborioso de ingeniería de características.

![[Pasted image 20240123151139.png]]

### Transición a Modelos No Paramétricos
Estamos experimentando una transición de descripciones estadísticas paramétricas a modelos completamente no parametrizados  donde los datos son escasos, es necesario confiar en la simplificación de suposiciones sobre la realidad para obtener modelos útiles. Cuando los datos son abundantes, estos pueden ser reemplazados por modelos no paramétricos que se ajusten a la realidad con mayor precisión. En cierta medida, esto refleja el progreso que experimentó la física a mediados del siglo anterior con la disponibilidad de computadoras. En lugar de resolver aproximaciones paramétricas de cómo se comportan los electrones a mano, ahora se puede recurrir a simulaciones numéricas de las ecuaciones diferenciales parciales asociadas. Esto ha llevado a modelos mucho más precisos, aunque a menudo a expensas de la explicabilidad.

![[Pasted image 20240123152150.png]]

