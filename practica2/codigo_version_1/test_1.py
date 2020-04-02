#pytest -q tests/test_*.py

import pytest
import api_v1

def test_api_v1_1():
    n = 2
    salida = api_v1.modelo_arima_completo_construido(n, './modeloT.pckl', './modeloH.pckl')
    assert  len(salida) == 2

def test_api_v1_2():
    n = 2
    modelo_v1_t = api_v1.recuperar_modelo('./modeloT.pckl')
    modelo_v1_h = api_v1.recuperar_modelo('./modeloH.pckl')
    pred_v1_t = api_v1.funcion_predecir_v1(modelo_v1_t, n )
    pred_v1_h = api_v1.funcion_predecir_v1(modelo_v1_h, n) 
    dias, temp, hum = api_v1.prediccion_to_diccionario(n, pred_v1_t, pred_v1_h)
    dic = api_v1.crear_diccionario_salida(dias, temp, hum)
    salida = api_v1.modelo_arima_completo_construido(n, './modeloT.pckl', './modeloH.pckl')
    assert  dic == salida
    assert len(dic) == len(salida)

