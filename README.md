![alt text](https://github.com/c4ttivo/MLOpsTaller1/blob/main/mlopstaller1/imgs/logo.png?raw=true)

# MLOps - Proyecto Final
## Autores
*    Daniel Crovo (dcrovo@javeriana.edu.co)
*    Carlos Trujillo (ca.trujillo@javeriana.edu.co)

## Profesor
*    Cristian Diaz (diaz.cristian@javeriana.edu.co)

## Arquitectura

![alt text](https://github.com/dcrovo/MLOPS-Final-Project/blob/main/img/architecture.png?raw=true)

### Descripción de componentes

La solución está compuesta por los siguientes contenedores:

*	**Airflow**: Orquestador para gestionar y programar los flujos de trabajo, relacionados con la recolección de datos, entrenamiento de modelos y registro de experimentos en MLFlow.
*	**Mlflow**: Registro de experimentos, seguimiento de métricas y almacenamiento de modelos. Configurado para usar **Minio** para el almacenamiento de objetos y **Postgresql** como base de datos para la metadata.
*	**Minio**: Almacen de objetos compatible con S3.
*	**Postgresql**: Se encuentran cuatro contenedores, uno de apoyo a Airflow, otro de apoyoa a MLflow y otros dos para el almacenamiento de la recolección de datos.
*	**Inference**: Servicio de FastAPI que consume el modelo entrenado y almacenado en MLflow y que permite hacer inferencias.
*	**Streamlit**: Aplicación que expone una interfaz web para la realización de inferencias.

## Requisitos

La solución requiere que se tenga una cuenta creada en DockerHub que será usada para almacenar las imágenes construidas con GitHub Actions. Así mismo se requiere que el repositorio de GitHub de donde se descarga el proyecto para su despliegue tenga la siguiente configuración:

1.    Vaya al repositorio de GitHub
2.    Navegue a través de "Settings > Secrets and variables > Actions".
3.    Agregue los siguientes secrets:

	-    "DOCKERHUB_USERNAME": El nombre de usuario de DockerHub
	-    "DOCKERHUB_PASSWORD": El password del usuario de DockerHub

Lo anterior va a garantizar que GitHub Actions tenga acceso a DockerHub para publicar las imagenes que se requieren.

**NOTA**: Si está usando classic tokens para la publicación de contenido en el repositorio debe asegurarse de que tenga los permisos adecuados para ejecutar workflows.


## Instrucciones
Clone el repositorio de git usando el siguiente comando en la consola de su sistema operativo:


```
# git clone https://github.com/dcrovo/MLOPS-Final-Project.git
```

Una vez ejecutado el comando anterior aparece el folder MLOPS-Final-Project. Luego es necesario ubicarse en el directorio de trabajo en el que se encuentra el archivo docker-compose.yml.


```
# cd MLOPS-Final-Project/
```

Ahora es necesario desplegar los contenedores


```
# sudo docker-compose up
```
En este paso se descarga las imágenes de acuerdo con lo especificado en el archivo docker-compose.yml.

<img src="https://github.com/dcrovo/MLOPS-Final-Project/blob/main/img/console.png?raw=true" width="50%" height="50%" />

Una vez finalizada la creación de los contenedores, se debe poder ingresar a las aplicaciones de cada contenedor a través de las siguientes URLs:

http://10.43.101.155:8083/ </br>
<img src="https://github.com/dcrovo/MLOPS-Final-Project/blob/main/img/minio.png?raw=true" width="50%" height="50%" /> </br>
http://10.43.101.155:8082/ </br>
<img src="https://github.com/dcrovo/MLOPS-Final-Project/blob/main/img/mlflow.png?raw=true" width="50%" height="50%" /> </br>
http://10.43.101.155:8080/ </br>
<img src="https://github.com/dcrovo/MLOPS-Final-Project/blob/main/img/airflow.png?raw=true" width="50%" height="50%" /> </br>

## Configuración

Los siguientes pasos permiten realizar la configuración del ambiente luego de ser desplegado.

1.	Dado que la VM está en Linux, se requiere establecer la siguiente variable.
```
	# echo -e "AIRFLOW_UID=$(id -u)" > .env
```	
2.	A continuación se debe configurar el bucket de S3, con el nombre **mlflows3** requerido por **MLflow**.

## Predicción

A través de la interfaz de Streamlit, es posible hacer predicciones usando el modelo almacenado y etiquetado @produccion.

http://10.43.101.155:8089 </br>

![alt text](https://github.com/dcrovo/MLOPS-Final-Project/blob/main/img/inference.png?raw=true)


## Video


[![IMAGE ALT TEXT HERE]([https://img.youtube.com/vi/4flOEZq96F0/0.jpg)](https://www.youtube.com/watch?v=4flOEZq96F0](https://youtu.be/Y-5fUG5H-Zw))


