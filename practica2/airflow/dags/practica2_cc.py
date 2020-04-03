#Import necesarios
from datetime import timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import requests

import pandas as pd
import pmdarima as pm
import numpy as np
import pickle

#Argumentos por defecto
default_args = {
    #Propietario Ángel Murcia Díaz
    'owner': 'Angel Murcia Diaz',
    'depends_on_past': False,
    #Fecha de inicio
    #Empezo hace dos dias, por tanto empieza siempre cuando se ejecuta
    'start_date': days_ago(2),
    'email': ['angel.ns.333@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    #'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

#Inicialización del grafo DAG de tareas para el flujo de trabajo
dag = DAG(
    'practica2_cc',
    default_args=default_args,
    description='Despliegue de un servicio Cloud Native para la predicción de humedad y temperatura.',
    #Cada cuanto se ejecuta
    schedule_interval=None,
)

# Funciones en Python

def guardar_modelo(modelo, ruta):
    fichero = open(ruta ,'wb')
    pickle.dump(modelo, fichero) 
    fichero.close()

def funcion_modelo_v1_t(**kwargs):
    df = pd.read_csv( kwargs['ruta_csv'] , header=0)
    print(df.head())
    df = df.dropna()
    modelo = pm.auto_arima(df.TEMP, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)
    guardar_modelo(modelo, kwargs['ruta_guardar_modelo'] )
    return modelo

def funcion_modelo_v1_h(**kwargs):
    df = pd.read_csv( kwargs['ruta_csv'] , header=0)
    print(df.head())
    df = df.dropna()
    modelo = pm.auto_arima(df.HUM, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)
    guardar_modelo(modelo, kwargs['ruta_guardar_modelo'] )
    return modelo




# Operadores o tareas

#PrepararEntorno: hace lo necesario para empezara trbajar en mi casoo crear un directorio en los ficheros temporales
PrepararEntorno = BashOperator(
    task_id='preparar_entorno',
    depends_on_past=False,
    bash_command='mkdir -p /tmp/workflow/v1',
    dag=dag,
)

#PrepararEntorno2: hace lo necesario para empezara trbajar en mi casoo crear un directorio en los ficheros temporales
PrepararEntorno2 = BashOperator(
    task_id='preparar_entorno_2',
    depends_on_past=False,
    bash_command='mkdir -p /tmp/workflow/v2',
    dag=dag,
)

#DatosA: descarga de la primera fuente de datos (humedad) desde github
DatosA = BashOperator(
    task_id='descarga_datos_Humedad',
    depends_on_past=True,
    bash_command='curl -o /tmp/workflow/humidity.csv.zip https://raw.githubusercontent.com/manuparra/MaterialCC2020/master/humidity.csv.zip',
    dag=dag,
)

#DatosB: descarga de la segunda fuente de datos (temperatura) desde github
DatosB = BashOperator(
    task_id='descarga_datos_Temperatura',
    depends_on_past=True,
    bash_command='curl -o /tmp/workflow/temperature.csv.zip https://raw.githubusercontent.com/manuparra/MaterialCC2020/master/temperature.csv.zip',
    dag=dag,
)

#DescomprimirA: descomprime el fichero de humedad anteriormente descargado
DescomprimirA = BashOperator(
    task_id='descomprimir_datos_Humedad',
    depends_on_past=True,
    bash_command='unzip -od /tmp/workflow/ /tmp/workflow/humidity.csv.zip',
    dag=dag,
)

#DescomprimirA: descomprime el fichero de temperatura anteriormente descargado
DescomprimirB = BashOperator(
    task_id='descomprimir_datos_Temperatura',
    depends_on_past=True,
    bash_command='unzip -od /tmp/workflow/ /tmp/workflow/temperature.csv.zip',
    dag=dag,
)

#LimpiarA: deja el conjunto de datos con tan solo 2 columnas la del tiempo y la de SanFrancisco
LimpiarA = BashOperator(
    task_id='limpiar_datos_Humedad',
    depends_on_past=True,
    bash_command='cut -d "," -f 1,4 /tmp/workflow/humidity.csv >> /tmp/workflow/humidity-clean.csv',
    dag=dag,
)

#LimpiarB: deja el conjunto de datos con tan solo 2 columnas la del tiempo y la de SanFrancisco
LimpiarB = BashOperator(
    task_id='limpiar_datos_Temperatura',
    depends_on_past=True,
    bash_command='cut -d "," -f 1,4 /tmp/workflow/temperature.csv >> /tmp/workflow/temperature-clean.csv',
    dag=dag,
)

#UnirAB: une los dos dataset con el campo del tiempo en comun (por defecto 1er campo)
UnirAB = BashOperator(
    task_id='unir_temperatura_Humedad',
    depends_on_past=True,
    bash_command='join -t "," /tmp/workflow/humidity-clean.csv /tmp/workflow/temperature-clean.csv >> /tmp/workflow/dataset_unido.csv',
    dag=dag,
)

#RetocarAB: pone a punto el dataset unificado, poniendo de separacion; y cambiando la primera linea para obtener las etiquetas deseadas
RetocarAB = BashOperator(
    task_id='poner_a_punto_Dataset',
    depends_on_past=True,
    bash_command="sed 's/datetime,San Francisco,San Francisco/DATE,HUM,TEMP/g' /tmp/workflow/dataset_unido.csv  >> /tmp/workflow/dataset_final.csv",
    dag=dag,
)

#CrearImagenBD: descarga la latest imagen de MariaDB de docker 
DescargarImagenBD = BashOperator(
    task_id='descargar_imagen_Mongo',
    depends_on_past=True,
    bash_command="docker pull mongo:latest",
    dag=dag,
)

#CrearContenedorBD: levanta (run) el contenedor con la imagen anteriormente descargada
CrearContenedorBD = BashOperator(
    task_id='crear_contenedor_Mongo',
    depends_on_past=True,
    bash_command="docker run -d -p 28900:27017 mongo:latest",
    dag=dag,
)



#ImportarDatosBD: importa los datos del fichero csv
ImportarDatosBD = BashOperator(
    task_id='importar_datos_a_BD_Contenedor',
    depends_on_past=True,
    bash_command="mongoimport -d BD1 -c sanFrancisco --file /tmp/workflow/dataset_final.csv --type csv --drop --port 28900 --headerline --host localhost",
    dag=dag,
)

#ImportarDatosBD: importa los datos del fichero csv
ExportarDatosBD = BashOperator(
    task_id='exportar_datos_a_CSV',
    depends_on_past=True,
    bash_command="mongoexport -d BD1 -c sanFrancisco --out /tmp/workflow/dataset_desde_mongo.csv --forceTableScan  --port 28900 --host localhost --type csv -f DATE,HUM,TEMP",
    dag=dag,
)

#ImportarDatosBD: importa los datos del fichero csv
RecortarBD = BashOperator(
    task_id='recortar_dataset',
    depends_on_past=True,
    bash_command="head -n 1000 /tmp/workflow/dataset_desde_mongo.csv > /tmp/workflow/dataset_recortado.csv",
    dag=dag,
)

#ModeloArimaTemperatura: realiza y guarda en pickle el modelo de la temperatura
ModeloArimaTemperatura = PythonOperator(
    task_id='crear_guardar_modelo_arima_temperatura',
    provide_context=True,
    python_callable=funcion_modelo_v1_t,
    op_kwargs={
        'ruta_csv': '/tmp/workflow/dataset_recortado.csv',
        'ruta_guardar_modelo': '/tmp/workflow/v1/modeloT.pckl'
    },
    dag=dag,
)

#ModeloArimaHumedad: realiza y guarda en pickle el modelo de la humedad
ModeloArimaHumedad = PythonOperator(
    task_id='crear_guardar_modelo_arima_humedad',
    provide_context=True,
    python_callable=funcion_modelo_v1_h,
    op_kwargs={
        'ruta_csv': '/tmp/workflow/dataset_recortado.csv',
        'ruta_guardar_modelo': '/tmp/workflow/v1/modeloH.pckl'
    },
    dag=dag,
)

#DescargarRequirementsV1: descarga de mi repositorio de github el requirements de la v1
DescargarRequirementsV1 = BashOperator(
    task_id='descargar_requirements_v1',
    depends_on_past=True,
    bash_command='curl -o /tmp/workflow/v1/requirements.txt https://raw.githubusercontent.com/NSInductus/CC2_UGR/master/practica2/codigo_version_1/requirements.txt',
    dag=dag,
)

#DescargarApiV1 : descarga de mi repositorio de github el la api de la v1
DescargarApiV1 = BashOperator(
    task_id='descargar_api_v1',
    depends_on_past=True,
    bash_command='curl -o /tmp/workflow/v1/api_v1.py https://raw.githubusercontent.com/NSInductus/CC2_UGR/master/practica2/codigo_version_1/api_v1.py',
    dag=dag,
)
#DescargarDockerfileV1: descarga de mi repositorio de github el dockerfile de la v1
DescargarDockerfileV1 = BashOperator(
    task_id='descargar_dockerfile_v1',
    depends_on_past=True,
    bash_command='curl -o /tmp/workflow/v1/Dockerfile https://raw.githubusercontent.com/NSInductus/CC2_UGR/master/practica2/codigo_version_1/Dockerfile',
    dag=dag,
)

#DescargarTestV1: descarga de mi repositorio de github el test de la v1
DescargarTestV1 = BashOperator(
    task_id='descargar_test_v1',
    depends_on_past=True,
    bash_command='curl -o /tmp/workflow/v1/test_1.py https://raw.githubusercontent.com/NSInductus/CC2_UGR/master/practica2/codigo_version_1/test_1.py',
    dag=dag,
)

#TestearApiV1: testear la api de la v1
TestearApiV1 = BashOperator(
    task_id='testear_api_v1',
    depends_on_past=True,
    bash_command='pytest /tmp/workflow/v1/ -p no:warnings',
    dag=dag,
)

#CrearImagenV1: crea la imagen de docker del servicio v1
CrearImagenV1 = BashOperator(
    task_id='crear_imagen_docker_v1',
    depends_on_past=True,
    bash_command='docker build -t imagen_1 /tmp/workflow/v1/',
    dag=dag,
)

#ArrancarDockerfileV1: aranca el dockerfile con la imagen (v1) que hemos creado posteriormente
ArrancarDockerfileV1 = BashOperator(
    task_id='arrancar_dockerfile_v1',
    depends_on_past=True,
    bash_command='docker run -d -p 8000:8000 imagen_1',
    dag=dag,
)

#head -2000 /tmp/workflow/dataset_final.csv >> /tmp/workflow/dataset_rapido.csv

#DescargarRequirementsV2: descarga de mi repositorio de github el requirements de la v2
DescargarRequirementsV2 = BashOperator(
    task_id='descargar_requirements_v2',
    depends_on_past=True,
    bash_command='curl -o /tmp/workflow/v2/requirements.txt https://raw.githubusercontent.com/NSInductus/CC2_UGR/master/practica2/codigo_version_2/requirements.txt',
    dag=dag,
)

#DescargarApiV2 : descarga de mi repositorio de github el la api de la v2
DescargarApiV2 = BashOperator(
    task_id='descargar_api_v2',
    depends_on_past=True,
    bash_command='curl -o /tmp/workflow/v2/api_v2.py https://raw.githubusercontent.com/NSInductus/CC2_UGR/master/practica2/codigo_version_2/api_v2.py',
    dag=dag,
)
#DescargarDockerfileV2: descarga de mi repositorio de github el dockerfile de la v2
DescargarDockerfileV2 = BashOperator(
    task_id='descargar_dockerfile_v2',
    depends_on_past=True,
    bash_command='curl -o /tmp/workflow/v2/Dockerfile https://raw.githubusercontent.com/NSInductus/CC2_UGR/master/practica2/codigo_version_2/Dockerfile',
    dag=dag,
)

#DescargarTestV2: descarga de mi repositorio de github el test de la v2
DescargarTestV2 = BashOperator(
    task_id='descargar_test_v2',
    depends_on_past=True,
    bash_command='curl -o /tmp/workflow/v2/test_2.py https://raw.githubusercontent.com/NSInductus/CC2_UGR/master/practica2/codigo_version_2/test_2.py',
    dag=dag,
)

#TestearApiV2: testear la api de la v2
TestearApiV2 = BashOperator(
    task_id='testear_api_v2',
    depends_on_past=True,
    bash_command='pytest /tmp/workflow/v2/ -p no:warnings',
    dag=dag,
)

#CrearImagenV2: crea la imagen de docker del servicio v2
CrearImagenV2 = BashOperator(
    task_id='crear_imagen_docker_v2',
    depends_on_past=True,
    bash_command='docker build -t imagen_2 /tmp/workflow/v2/',
    dag=dag,
)

#ArrancarDockerfileV2: aranca el dockerfile con la imagen (v2) que hemos creado posteriormente
ArrancarDockerfileV2 = BashOperator(
    task_id='arrancar_dockerfile_v2',
    depends_on_past=True,
    bash_command='docker run -d -p 8010:8010 imagen_2',
    dag=dag,
)



#cut -d "," -f 1 ./humidity.csv >> /tmp/workflow/nuevo.csv
#inductus@inductusPC:/tmp/workflow$ cut -d "," -f 1 ./humidity.csv >> /tmp/workflow/humidity-clean.csv
#inductus@inductusPC:/tmp/workflow$ cut -d "," -f 1,4 ./temperature.csv >> /tmp/workflow/temperature-clean.csv
#inductus@inductusPC:/tmp/workflow$ cut -d "," -f 1,4 ./humidity.csv >> /tmp/workflow/humidity-clean.csv
#inductus@inductusPC:/tmp/workflow$ cut -d "," -f 1,4 ./humidity.csv >> /tmp/workflow/humidity-clean.csv
#inductus@inductusPC:/tmp/workflow$ join -t "," ./humidity-clean.csv ./temperature-clean.csv >> /tmp/workflow/dataset.csv

#sed 's/,/;/g' ./dataset.csv | sed 's/datetime;San Francisco;San Francisco/DATE;TEMP;HUM/g' >> dataset_final.csv


#CreaVisualizacion = PythonOperator(
 #   task_id='extrae_mapa',
  #  provide_context=True,
   # python_callable=funcion nombre,
    #op_kwargs ={'url': ""}
    #dag=dag,
#)


#mongoimport -d nombreBDJ -c nombreColeccion --file /tmp/workflow/dataset_final.csv --type csv  --drop --port 27017 --headerline


#delimitador ; .... no funciona correctamente cmabiar en consecuencia

#docker run -p 28900:27017 97a9a3e85158 
#mongoimport -d new -c a --file /tmp/workflow/dataset_unido.csv --type csv  --drop --port 28900 -f DATE,HUM,TEMP --host localhost

#Dependencias
#t1 >> [t2, t3]
PrepararEntorno >> PrepararEntorno2
PrepararEntorno2 >> [DatosA , DatosB]
DatosA >> DescomprimirA
DatosB >> DescomprimirB
DescomprimirA >> LimpiarA
DescomprimirB >> LimpiarB
[LimpiarA , LimpiarB] >> UnirAB
UnirAB >> RetocarAB
RetocarAB >> DescargarImagenBD
DescargarImagenBD >> CrearContenedorBD
CrearContenedorBD >> ImportarDatosBD
ImportarDatosBD >> ExportarDatosBD
ExportarDatosBD >> RecortarBD
RecortarBD >> ModeloArimaTemperatura
RecortarBD >> ModeloArimaHumedad
ModeloArimaTemperatura >> DescargarApiV1
ModeloArimaHumedad >> DescargarApiV1
ModeloArimaTemperatura >> DescargarRequirementsV1
ModeloArimaHumedad >> DescargarRequirementsV1
ModeloArimaTemperatura >> DescargarDockerfileV1
ModeloArimaHumedad >> DescargarDockerfileV1
ModeloArimaTemperatura >> DescargarTestV1
ModeloArimaHumedad >> DescargarTestV1
DescargarApiV1 >> TestearApiV1
DescargarRequirementsV1 >> TestearApiV1
DescargarDockerfileV1 >> TestearApiV1
DescargarTestV1 >> TestearApiV1 
TestearApiV1 >> CrearImagenV1
CrearImagenV1 >> ArrancarDockerfileV1
PrepararEntorno2 >> DescargarApiV2
PrepararEntorno2 >> DescargarRequirementsV2
PrepararEntorno2 >> DescargarDockerfileV2
PrepararEntorno2 >> DescargarTestV2
DescargarApiV2 >> TestearApiV2
DescargarRequirementsV2 >> TestearApiV2
DescargarDockerfileV2 >> TestearApiV2
DescargarTestV2 >> TestearApiV2
TestearApiV2 >> CrearImagenV2
CrearImagenV2 >> ArrancarDockerfileV2

###Eliminar todo cvarpeta y docker
#lanzar el flujoa  ver si crea correctamente modelo
#con eso lanzar api rest y test a ver

#poner codigo y test en github y bajarlo con airflow

#Dockerfiule y historias


#la copia de modelos completos est aen el critorio prosi