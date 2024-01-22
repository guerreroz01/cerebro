# RAG - Retrieval Augmented Generation
Es un método en el cual se puede aumentar la información contextual de un modelo a la hora de generar respuestas 

![[RAG_paper.pdf|Paper]]

## Embeddings
Son representaciones vectoriales de palabras, de términos que tratan de retener parte del significado, de la semántica de las palabras que estamos convirtiendo en estos embedding. 


>[!tip] Existen generadores locales de embeddings

![[Pasted image 20240121172657.png]]

## Similitud coseno
La **similitud coseno** es una medida de la similitud que existe en un espacio que posee un producto interior con el que se evalúa el valor del coseno del ángulo comprendido entre ellos.

Esta función trigonométrica proporciona un valor igual a 1 si el ángulo comprendido es cero, es decir si ambos vectores apuntan a un mismo lugar. 

Cualquier ángulo existente entre los vectores, el coseno arrojaría un valor inferior a uno. Si los vectores fuesen ortogonales el coseno se anularía, y si apuntasen en sentido contrario su valor sería -1. De esta forma, el valor de esta métrica se encuentra entre -1 y 1, es decir en el intervalo cerrado [-1, 1].

Esta distancia se emplea frecuentemente en la búsqueda y recuperación de información representando las palabras (o documento) en un espacio vectorial. En minería de textos se aplica la similitud coseno con el objeto de establecer una métrica de semejanza entre textos. En minería de datos se suele emplear como un indicador de cohesión de clústeres de texto.

La similitud coseno no debe ser considerada como una métrica debido a que no cumple con la desigualdad triangular.
![[Pasted image 20240121173803.png]]


## Montaje de elementos 
![[Pasted image 20240121175004.png]]

Nuestra base de datos vectorial va a ser nuestro índice documental, va a ser donde guardemos y gestionemos todos los fragmentos de los documentos que queremos analizar, 

### Y esto que implica?
Que para cada uno de los documentos que queremos añadir vamos a tener que calcular los embeddings 

### Como influye esto en el proceso de generación 
![[Pasted image 20240121175253.png]]

Básicamente se tiene una consulta, voy a pasar esa consulta a mi modelo de lenguaje y me va a dar la respuesta. 

### Pero como interactua esto con el RAG? 
![[Pasted image 20240121175515.png]]

Imaginad que tenemos una conversación con un sistema de chat al que le vamos a dar unas instrucciones iniciales, una serie de pautas, una serie de ejemplos, y tenemos una consulta 1, nos responde el sistema, tenemos una consulta 2, nos responde el sistema, cuando nos llega la consulta 3 el contexto que tiene que evaluar nuestro gran modelo de lenguaje, no sólo es la consulta 3, sino que son todas las interacciones previas que hemos tenido.

Cuando estamos trabajando con un sistema que nos permite realizar distintas interacciones, es fundamental entender la _importancia del contexto_.

![[Pasted image 20240121180226.png]]
Cuando yo estoy haciendo mi proceso de consulta y cuando yo estoy obteniendo los datos, realmente lo que estoy haciendo es _inyectar_ información útil en ese contexto 

![[Pasted image 20240121180319.png]]
Si vemos como funciona _ChromaDB_ por ejemplo, tenemos una aplicación que hace unas consultas, las consultas se convierten en un [[#Embeddings]] y básicamente la base de datos va a devolver los "X" fragmentos que más se parezcan a los embeddings de la consulta.

Es decir yo estoy haciendo una búsqueda de similaridad, buscando lo que tengo en mi base de datos a que se parece más a la query, esos van a ser 1, 2, 3 fragmentos de unos 1000 tokens, y esos 1, 2, 3 fragmentos de 1000 tokens, los voy a meter en mi ventana de query como contexto.

Es decir, le voy a facilitar a mi sistema información contextualizada para la pregunta, _**antes de que nos de la respuesta**_, es decir la respuesta se va a ver aumentada por esta búsqueda de fragmentos similares.

Lo que en teoría debería reducir la posibilidad de que alucinara, si le doy un trozo de la Wikipedia donde le digo los nombres de los premios Nobel de 2023, es mucho menos posible que se invente el resultado, que si simplemente le pido esto sin suministrarle la información. 
![[Pasted image 20240121181208.png]]

![[Pasted image 20240121181343.png]]








# Referencias
____
1. Buscar el programa PanDoc que permite convertir cualquier documento a un archivo TXT
2. Los embeddings se pueden guardar en una base de datos vectorial como ChromaDB o similares pgvector