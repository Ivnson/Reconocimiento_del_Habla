"""
Normalizador canónico de texto para la Práctica 2 (ASR).

ESTE FICHERO NO SE MODIFICA.

Es la única fuente de verdad de la normalización. Se aplica por igual a la
hipótesis del modelo y a la transcripción de referencia. Cualquier cambio aquí
invalida la comparación entre equipos.

Criterios aplicados (ver apartado 4.1 de la guía):
  - normalización Unicode a NFC
  - minúsculas
  - conservación de tildes y de la eñe
  - numerales convertidos a palabras
  - eliminación de signos de puntuación
  - colapso de espacios en blanco
"""

import re
import unicodedata

from num2words import num2words

# Caracteres que sobreviven a la limpieza: letras del español, dígitos
# (ya convertidos en ese punto) y espacio.
_PERMITIDOS = re.compile(r"[^a-záéíóúüñ0-9 ]")

# Símbolos que se leen en voz alta y por tanto deben expandirse, no borrarse.
_SIMBOLOS = {
    "%": " por ciento ",
    "€": " euros ",
    "$": " dólares ",
    "&": " y ",
    "+": " más ",
    "=": " igual ",
    "°": " grados ",
    "º": " grados ",
}

# Guiones, barras y dos puntos separan palabras (y horas); el resto de la
# puntuación simplemente desaparece.
_SEPARADORES = re.compile(r"[-–—/:]")

_MILES = re.compile(r"(?<=\d)\.(?=\d{3}\b)")
_DECIMAL = re.compile(r"\b(\d+),(\d+)\b")
_ENTERO = re.compile(r"\b\d+\b")


def _entero_a_palabras(texto: str) -> str:
    return num2words(int(texto), lang="es")


def _decimal_a_palabras(m: re.Match) -> str:
    entera, decimal = m.group(1), m.group(2)
    # En español se lee "tres coma cinco", no "tres punto cinco".
    # La parte decimal se lee dígito a dígito cuando tiene ceros a la izquierda.
    if decimal.startswith("0"):
        cola = " ".join(num2words(int(d), lang="es") for d in decimal)
    else:
        cola = num2words(int(decimal), lang="es")
    return f"{num2words(int(entera), lang='es')} coma {cola}"


def normalizar(texto: str) -> str:
    """Devuelve la forma canónica de `texto` para el cálculo de WER y CER."""
    if texto is None:
        return ""

    # NFC primero: dos modelos pueden emitir la misma tilde como carácter
    # precompuesto o como letra + diacrítico combinante. Sin esto, "á" y "á"
    # cuentan como caracteres distintos y el CER sale inflado.
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.lower()

    for simbolo, palabra in _SIMBOLOS.items():
        texto = texto.replace(simbolo, palabra)

    # Los separadores pasan a espacio ANTES de tocar los números. Si no,
    # "23:15" se queda en "veintitrésquince" al caer los dos puntos entre
    # dos números ya expandidos. Como espacio, da "veintitrés quince", que es
    # como lo leería cualquiera.
    texto = _SEPARADORES.sub(" ", texto)

    # Separadores de millar fuera antes de interpretar los números.
    texto = _MILES.sub("", texto)
    texto = _DECIMAL.sub(_decimal_a_palabras, texto)
    texto = _ENTERO.sub(lambda m: _entero_a_palabras(m.group(0)), texto)

    texto = _PERMITIDOS.sub("", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


if __name__ == "__main__":
    pruebas = [
        "El tren de las 23:15 llegó con 3,5 minutos de retraso.",
        "¿Cuánto cuesta? 1.200 € —casi un 15 % más—.",
        "La señora Núñez vive en Aljaraque.",
    ]
    for p in pruebas:
        print(f"{p!r}\n  -> {normalizar(p)!r}")
