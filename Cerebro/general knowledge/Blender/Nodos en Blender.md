---
banner: general knowledge/Blender/assets/nodos_blender.png
---
<iframe width="auto" height="650" style="width: 800px" src="https://www.youtube.com/embed/vOFLhLtBlFE" title="Masterclass: Cómo usar los nodos de Blender (¡desde cero!)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
# Nodos:
___

Un nodo es un punto de intersección, conexión o unión de varios elementos que confluyen en el mismo lugar. [wikipedia](https://wikipedia.com)

![[Pasted image 20240408164430.png]]
En las entradas podemos introducir información.


## Tipos de nodos:
![[Pasted image 20240408164531.png]]
### Nodos de entrada:
Sólo tienen una salida, introducen información al árbol de nodos puede ser una imagen de textura, un vídeo etc.

### Nodos modificador:
Puede ser por ejemplo una corrección de color, cambiar los valores de los colores, un color ramp etc, un nodo modificar toma el valor de entrada y lo modifica para luego generar una salida.

### Nodos de salida:
Existen pocos, y generan un resultado, puede ser un nodo output, o el nodo de salida que es el resultado final.

### Nodos Mix
![[Pasted image 20240408165120.png]]
Sirven para mezclar valores,  colores,  incluso hacer operaciones matemáticas etc. 

Debido a que los Mix sólo mezclan dos elementos necesitamos otro Mix si queremos mezclar 3 o más elementos.
![[Pasted image 20240408165350.png]]

Los nodos Mix son muy útiles cuando se utilizan Máscaras.
![[Pasted image 20240408165735.png]]
El nodo mix tiene un parámetro que se llama factor en el cual se puede introducir una imagen en blanco y negro y esa imagen va a controlar que parte de cual de las entradas se va a ver, el factor suele ser un slider que va desde 0 a 1. Las partes negras tienen un valor de 0, el negro es 0 el blanco es 1, las máscaras son imágenes en escala de grises 

## Tipos de conexiones:
![[Pasted image 20240408170306.png]]
Las conexiones son esos puntos en cada nodo, los que están a la izquierda son entradas y los que están a la derecha son salidas. 

Estas entradas vienen de colores:
- El gris es un valor o número o factor.
- La entrada amarilla son colores en formato RGB.
- La entrada azul son vectores xyz, texturas en forma de direcciones.

>[!danger] La salida y la entrada tienen que ser del mismo tipo


## ¿Qué pueden hacer los Nodos?
- Crear materiales
- Crear o editar texturas
- Crear máscaras a partir de propiedades de un objeto
- Efecto de aleatoriedad y variación
- Mezclar diferentes componentes
- Optimización y limitación / control de efectos
- Composición y [[VFX]], [[partículas]]

## Diferencia entre textura y material:
El <span style="color:#ff0000">material</span> son las propiedades que tiene una superficie de un objeto, puede ser la reflexión, rugosidad, si es metálico o no, si es transparente o si es opaco, todas esas propiedades son parte del material y la manera de controlar esas propiedades se realiza a través de las <span style="color:#ffff00">texturas.</span>

Es decir a través de una textura podemos decirle a un material que en una zona se opaco y en otra zona no.

>[!warning] Una textura no hace nada si no la conectamos a un material.

 
## Estructura de nodos básica
___
Dos tipos de nodos:
- Imágenes / colores
- Shaders