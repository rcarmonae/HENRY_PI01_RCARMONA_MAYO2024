# HENRY_PI01_RCARMONA_MAYO2024
Proyecto Individual #1 de Machine Learning Operations HENRY.

El proyecto consiste en un proyecto de práctica cuyo propósito es aplicar métodos y técnicas de Data Engineering y Machine Learning a una base de datos de videojuegos de Steam como parte de la carrera de Ciencia de Datos de HENRY. El proyecto se organiza en dos fases: 1. Procesamiento de los datos y creación de una API y 2) Análisis Exploratorio de Datos (EDA) y 3) Creación de un modelo Machine Learning para hacer recomendaciones. 

A continuación, se describen los procesos y archivos correspondientes a cada fase, los cuales se encuentran en este repositorio:

## 1) Procesamiento de los datos y creación de una API
Consistió en la extracción y transformación de las base de datos originales provistas por HENRY. A partir de estas bases de datos se extrae información archivos json para generar nuevos archivos en formato parquet que se utilizan posteriormente en los EndPoints de la API, el EDA y el Sistema de Recomendación. Durante esta etapa también se construyó la aplicación utilizando las herramientas FastAPI y Render. La liga de acceso a la aplicación es la siguiente: https://henry-pi01-rcarmona-mayo2024.onrender.com/docs. Los archivos parquet que se enlistan en esta sección son los que alimentan todas las funciones que se pueden consultar mediante la API en render.

### Archivos:
DATASETS:
 - F01_PlaytimeGenre.parquet - Contiene la base de datos necesaria para ejecutar la primera función. NOTA: La base de datos que se genera a partir de los datos orginales (provistos por HENRY) posee un total de 9,993,947 de filas. Por lo tanto, se tomó la decisión de crear una muestra aleatoria con el 1% de los datos con el fin de prevenir problemas relacionados con el uso de memoria. De esta  manera  se logra probar  la correcta ejecución de funciones y deploy.
 - F02_UserForGenre.parquet - Contiene la base de datos necesaria para ejecutar la segunda función. NOTA: La base de datos que se genera a partir de los datos orginales (provistos por HENRY) posee un total de 9,993,947 de filas. Por lo tanto, se tomó la decisión de crear una muestra aleatoria con el 1% de los datos con el fin de prevenir problemas relacionados con el uso de memoria. De esta  manera  se logra probar  la correcta ejecución de funciones y deploy.
 - F03_UsersRecommend.parquet - Contiene la base de datos necesaria para ejecutar la tercera función y el sistema de recomendación.
CODE:
 - EDA.ipynb - Contiene el análisis exploratorio de algunas variables de los datasets.
 - ETL.ipynb -  Contiene todas las transformaciones aplicadas a los datasets.
 - funciones.ipynb - Contiene un respaldo de las funciones que se incluyen en la API.
- main.py - Contiene las funciones de la API.
- requirements.txt - Contiene las dependencias de Python que este proyecto necesita para funcionar correctamente.

## 2) Creación de un modelo Machine Learning para hacer recomendaciones
Se eligió el modelo SIMILITUD DE COSENO para determinar la similitud de los videojuegos a partir de sus valores de género, año de lanzamiento, analisis de sentimiento a partir de las reseñas de usuarios y la recomendación positiva de los usuarios. 

