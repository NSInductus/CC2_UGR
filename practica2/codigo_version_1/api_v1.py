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

from statsmodels.tsa.arima_model import ARIMA

import requests
from datetime import datetime, timedelta, date

import pickle

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

def recuperar_modelo(ruta):
    fichero = open(ruta,'rb') 
    modelo = pickle.load(fichero)
    print(modelo)
    fichero.close()
    return modelo

def funcion_predecir_v1(model, n):
    # Forecast
    n_periods = n # One day
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    # fc contains the forecasting for the next 24 hours.
    #print(fc)
    return fc

def modelo_arima_completo_construido(n, rutaT, rutaH):
    modelo_v1_t = recuperar_modelo(rutaT)
    modelo_v1_h = recuperar_modelo(rutaH)
    pred_v1_t = funcion_predecir_v1(modelo_v1_t, n )
    pred_v1_h = funcion_predecir_v1(modelo_v1_h, n) 
    dias, temp, hum = prediccion_to_diccionario(n, pred_v1_t, pred_v1_h)
    dic = crear_diccionario_salida(dias, temp, hum)
    print(dic)
    return dic

app = Flask(__name__)


@app.route('/servicio/v1/<int:n>horas/', methods=['GET'])
def predecir_v1_rest(n):
    salida = modelo_arima_completo_construido(n, './modeloT.pckl', './modeloH.pckl')
    return Response(json.dumps(salida), status=200, mimetype="application/json")

