Cormack y Hunsfield introdujeron la tomografía computarizada en el año 1971

En 1998, Mozzo et al introdujeron un nuevo tipo de tomografía computarizada llamada tomografía computarizada de haz cónico (Cone Beam Computed Tomography, CBCT). Este sistema reduce el precio y la radiación si se compara con el TAC médico (helicoidal).

El desarrollo de los CBCT para el uso de la odontología comenzó en la segunda mitad de los años 90. Por lo tanto, pronto el uso del CBCT en odontología, cirugía maxilofacial y en otorrinolaringología se generalizó.

| CT | CBCT |
| ---- | ---- |
| CCD detector | CCD detector or plane screen |
| Haz plano | Haz cónico |
| Rotaciones múltiples (En cada rotación una imágen) | Una rotación (1-2 imágenes por grado) |
| Voxel anisotrópico (distinto en todas las direcciones del espacio) | Voxel isotrópico (igual en todas las direcciones del espacio) |
| Cortes de 1mm de espesor | Cortes de menos de 1 mm de espesor |
| Dosis de radiación alta | Dosis de radiación baja |

![[Pasted image 20240117123658.png]]

## Procesamiento de los datos e interpretación
1. Hardware de imágen (exposición a los rayos X)
2. Adquisición de la imagen (proyecciones tomadas por el detector que pueden reconstruirse en una representación 3D del objeto escaneado).
3. Reconstrucción de la imagen (pre-procesamiento de los datos no procesados). El software hace el procesamiento y la interpretación de los datos a través de pixels y voxels.

- Un pixel: es un elemento cuadrado, plano.
- Un voxel: es un cubo, un elemento con volumen.
![[Pasted image 20240117124037.png]]

Después de la adquisición de la imagen, ocurre la transformación a un archivo de imagen digital y de comunicación en medicina (digital imaging and communications in medicine, archivo DICOM) a través de la reconstrucción de algoritmos. Esta reconstrucción de imagen es una técnica para reconstruir una imagen a partir de proyecciones múltiples, la imagen reconstruida representa la atenuación relativa de los rayos X de los diferentes materiales en el objeto escaneado.

En el CBCT, el objeto escaneado es reconstruido en una matriz de voxels, en donde a cada voxel se le asigna un valor de gris dependiendo de la atenuación del material.

Por lo tanto, los pixeles y los voxels tienen un valor de brillo que puede ser representado en una escala de grises (HUNSFIELD).

Cada voxel está compuesto por 6 pixels con un valor de brillo que representa la densidad de la estructura.

Cada pixel representa un valor de atenuación de la radiación reprentado en la escala de Hounsfield
- -1000 aire
- 0 agua
- +3000 metal pesado

El ojo humano no puede distinguir 4000 tonos de grises, por esta razón, el software representa en la pantalla un rango de 256 tonos a través de un proceso llamado windowing. Los tonos por debajo de la Ventana se interpretarán como negro, y los que queden por encima de la ventana como blanco. Los archivos DICOM pueden exportarse a diferentes softwares.

![[Pasted image 20240117124933.png]]

