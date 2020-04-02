#FLASK_APP=Portatiles_rest.py flask run
#export FLASK_RUN_PORT=8000

from flask import Flask
from flask import request
from flask import Response

import pandas as pd
import pmdarima as pm
import numpy as np

#from bson import json_util
import json

#from statsmodels.tsa.arima_model import ARIMA
import requests
from datetime import datetime, timedelta, date

#import pickle

def funcion_v2(n):
    n=int(n/1)
    weatherJson= r = requests.get(url = "http://api.openweathermap.org/data/2.5/forecast?id=5391959&appid=c9d7622157c6e3c08780e4661832609b")
    data = weatherJson.json() 
    df = pd.io.json.json_normalize(data['list'])
    todays_date = datetime.now()
    index = pd.date_range(todays_date, periods=n, freq='1H')
    odf= pd.DataFrame(index=index, columns=['TEMP','HUM'])
    odf['TEMP']=df['main.temp'].head(n).values
    odf['HUM']=df['main.humidity'].head(n).values
    return odf

def crear_lista_proximos_dias(n):
    lista_dias = list()
    for i in range(n):
        today = datetime.today() 
        dias = timedelta(hours=i)
        tomorrow = today + dias
        #print(tomorrow)
        final= str(tomorrow.year)+ "-"+str(tomorrow.month)+ "-"+ str(tomorrow.day)+ " " + str(tomorrow.hour) + ":" + str(tomorrow.minute)
        lista_dias.append(final)
    return lista_dias 

def prediccion_to_diccionario(n, pred_t, pred_h):
    dias = list()
    dias = crear_lista_proximos_dias(n)
    temp = list(pred_t)
    hum = list(pred_h)
    return dias, temp, hum


def crear_diccionario_salida(dias, temp, hum):
    total = list()
    for i in range(len(dias)):
        dic = {'Fecha' : dias[i], 'Temperatura' : temp[i], 'Humedad': hum[i] }
        total.append(dic)
    return total

def modelo_web_completo(n):
    pred_v2 = funcion_v2( n )
    dias, temp, hum = prediccion_to_diccionario(n, pred_v2.TEMP, pred_v2.HUM)
    dic = crear_diccionario_salida(dias, temp, hum)
    print(dic)
    return dic

app = Flask(__name__)

@app.route('/servicio/v2/<int:n>horas/', methods=['GET'])
def predecir_v2_rest(n):
    salida = modelo_web_completo(n)
    return Response(json.dumps(salida), status=200, mimetype="application/json")

