#pytest -q tests/test_*.py

import pytest
import api_v2

def test_api_v2_1():
    n = 2
    salida = api_v2.funcion_v2(n)
    assert  len(salida) == 2 == n

def test_api_v2_2():
    n = 2
    salida1 = api_v2.funcion_v2(n)
    salida2 = api_v2.funcion_v2(n)
    assert len(salida1) == len(salida2)
