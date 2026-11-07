#!/usr/bin/env python3
"""
Comprueba que la entrega de la Practica 2 esta completa.

Uso:   python3 comprobar_entrega.py

No evalua la CALIDAD del trabajo, solo que no falta nada y que los numeros
son internamente coherentes. Pasar esta comprobacion es condicion necesaria
para que la entrega se corrija, no suficiente para aprobarla.

Lo que NO comprueba, porque requiere criterio: si los tres modelos son
realmente de familias distintas, y si la justificacion del informe esta bien
argumentada. Eso se valora en la defensa.
"""

import ast
import csv
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parent
OK, FALLO, AVISO = "[ OK ]", "[FALTA]", "[AVISO]"

fallos = 0
avisos = 0

CAMPOS = [
    "equipo", "modelo", "backend", "precision", "condicion", "entorno",
    "wer", "cer", "rtf_medio", "t_carga_s", "vram_pico_mb", "licencia",
]
CONDICIONES = ["limpio", "snr20", "snr10", "snr5", "telefono", "codec"]
COLS_RENDIMIENTO = ["rtf_medio", "t_carga_s", "vram_pico_mb"]
MIN_MODELOS = 3


def exige(ruta, descripcion):
    global fallos
    p = RAIZ / ruta
    if p.exists():
        print(f"{OK} {descripcion}")
        return p
    fallos += 1
    print(f"{FALLO} {descripcion}  ->  {ruta}")
    return None


def falla(mensaje):
    global fallos
    fallos += 1
    print(f"{FALLO} {mensaje}")


def sugiere(cond, mensaje):
    global avisos
    if not cond:
        avisos += 1
        print(f"{AVISO} {mensaje}")


# ---------------------------------------------------------------------------

print("COMPROBACION DE LA ENTREGA - Practica 2")
print()

exige("informe/informe.pdf", "Informe en PDF")
exige("p2_plantilla.ipynb", "Cuaderno de trabajo")

# --- El cuaderno, ejecutado ------------------------------------------------
nb_path = RAIZ / "p2_plantilla.ipynb"
if nb_path.exists():
    import json

    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        codigo = [c for c in nb["cells"] if c["cell_type"] == "code"]
        sin_ejecutar = [c for c in codigo if not c.get("outputs")
                        and "".join(c["source"]).strip()]
        sugiere(len(sin_ejecutar) <= len(codigo) // 3,
                f"{len(sin_ejecutar)} de {len(codigo)} celdas de codigo sin salida: "
                "se entrega con TODAS las celdas ejecutadas")
    except Exception as e:
        falla(f"el cuaderno no se puede leer: {type(e).__name__}: {e}")

# --- resultados.csv --------------------------------------------------------
filas = []
csv_path = RAIZ / "resultados" / "resultados.csv"
if not csv_path.exists():
    falla("no existe resultados/resultados.csv")
else:
    with csv_path.open(encoding="utf-8") as f:
        lector = csv.DictReader(f)
        cabecera = lector.fieldnames or []
        filas = list(lector)

    if cabecera != CAMPOS:
        falla("la cabecera del CSV no coincide con el esquema fijo\n"
              f"        esperada: {','.join(CAMPOS)}\n"
              f"        recibida: {','.join(cabecera)}")
        filas = []
    elif not filas:
        falla("resultados.csv solo tiene cabecera, sin datos")
    else:
        print(f"{OK} resultados.csv con {len(filas)} filas")

# --- Cobertura de modelos y condiciones ------------------------------------
if filas:
    modelos = sorted({f["modelo"].strip() for f in filas if f["modelo"].strip()})
    if len(modelos) < MIN_MODELOS:
        falla(f"hay {len(modelos)} modelo(s) distinto(s); se piden al menos "
              f"{MIN_MODELOS}, de familias arquitectonicas distintas")
    else:
        print(f"{OK} {len(modelos)} modelos: {', '.join(modelos)}")

    for modelo in modelos:
        suyas = {f["condicion"].strip() for f in filas if f["modelo"].strip() == modelo}
        faltan = [c for c in CONDICIONES if c not in suyas]
        if faltan:
            falla(f"a '{modelo}' le faltan condiciones: {', '.join(faltan)}")
        sobran = suyas - set(CONDICIONES)
        if sobran:
            falla(f"'{modelo}' tiene condiciones desconocidas: {', '.join(sorted(sobran))}")

    sugiere(len(filas) >= MIN_MODELOS * len(CONDICIONES),
            f"{len(filas)} filas; lo esperable son al menos "
            f"{MIN_MODELOS * len(CONDICIONES)} (modelos x condiciones)")

# --- La regla del rendimiento: se mide en NIMROD ---------------------------
if filas:
    for modelo in sorted({f["modelo"].strip() for f in filas if f["modelo"].strip()}):
        referencia = [
            f for f in filas
            if f["modelo"].strip() == modelo
            and f["entorno"].strip() == "nimrod"
            and f["condicion"].strip() == "limpio"
            and all(f[c].strip() for c in COLS_RENDIMIENTO)
        ]
        if not referencia:
            falla(f"'{modelo}' no tiene fila de rendimiento: hace falta una con "
                  "entorno=nimrod, condicion=limpio y rtf_medio, t_carga_s y "
                  "vram_pico_mb rellenos")

    for i, f in enumerate(filas, start=2):
        entorno = f["entorno"].strip()
        if entorno not in ("nimrod", "local"):
            falla(f"fila {i}: entorno '{entorno}' no valido (nimrod | local)")
        elif entorno == "local":
            puestas = [c for c in COLS_RENDIMIENTO if f[c].strip()]
            if puestas:
                falla(f"fila {i} ({f['modelo']}/{f['condicion']}): entorno=local "
                      f"pero {', '.join(puestas)} tiene valor. Esas columnas van "
                      "VACIAS fuera de NIMROD; no se estiman ni se copian")

# --- Coherencia de los numeros ---------------------------------------------
if filas:
    for i, f in enumerate(filas, start=2):
        etq = f"fila {i} ({f['modelo']}/{f['condicion']})"

        for col in ("wer", "cer"):
            crudo = f[col].strip()
            if not crudo:
                falla(f"{etq}: {col} vacio")
                continue
            try:
                v = float(crudo)
            except ValueError:
                falla(f"{etq}: {col}='{crudo}' no es un numero")
                continue
            if v < 0:
                falla(f"{etq}: {col}={v} es negativo")
            else:
                sugiere(v <= 1.5, f"{etq}: {col}={v}. Un WER por encima de 1 es "
                                  "posible (inserciones), pero revisad que no "
                                  "esteis dividiendo mal")

        for col in COLS_RENDIMIENTO:
            crudo = f[col].strip()
            if not crudo:
                continue
            try:
                v = float(crudo)
            except ValueError:
                falla(f"{etq}: {col}='{crudo}' no es un numero")
                continue
            if v <= 0:
                falla(f"{etq}: {col}={v} deberia ser positivo")
            elif col == "rtf_medio":
                sugiere(v <= 1.0,
                        f"{etq}: RTF={v} (mas lento que tiempo real). "
                        "Descartasteis las pasadas de calentamiento? "
                        "La primera inferencia cuesta hasta 26x mas")

        for col in ("equipo", "backend", "precision", "licencia"):
            if not f[col].strip():
                falla(f"{etq}: {col} vacio")

    # Un mismo modelo medido con dos backends distintos no es comparable.
    por_modelo = {}
    for f in filas:
        por_modelo.setdefault(f["modelo"].strip(), set()).add(f["backend"].strip())
    for modelo, backends in por_modelo.items():
        if len(backends) > 1:
            falla(f"'{modelo}' aparece con backends distintos "
                  f"({', '.join(sorted(backends))}). Son implementaciones "
                  "distintas: sus cifras no son comparables entre si")

# --- transcribe.py ---------------------------------------------------------
tr = RAIZ / "transcribe.py"
if not tr.exists():
    falla("no existe transcribe.py")
else:
    fuente = tr.read_text(encoding="utf-8")
    try:
        arbol = ast.parse(fuente)
    except SyntaxError as e:
        falla(f"transcribe.py no compila: linea {e.lineno}, {e.msg}")
        arbol = None

    if arbol is not None:
        fn = next((n for n in arbol.body
                   if isinstance(n, ast.FunctionDef) and n.name == "transcribe"), None)
        if fn is None:
            falla("transcribe.py no define una funcion transcribe() de nivel superior")
        else:
            print(f"{OK} transcribe() definida")
            if len(fn.args.args) != 1:
                falla(f"transcribe() recibe {len(fn.args.args)} argumentos; "
                      "el contrato es transcribe(ruta_audio) -> str")
            sugiere(ast.get_docstring(fn), "transcribe() no tiene docstring")

        sugiere("NotImplementedError" not in fuente,
                "transcribe.py aun contiene NotImplementedError: "
                "habeis implementado vuestro backend?")
        sugiere("lru_cache" in fuente or "_modelo" in fuente,
                "no veo cache del modelo en transcribe.py. La carga perezosa y "
                "cacheada es parte del contrato: P3 llamara a esta funcion en bucle")

# --- scripts/ sin tocar ----------------------------------------------------
# No se puede comprobar sin la version original, pero si avisar de lo obvio.
for nombre in ("normalize.py", "metrics.py", "asr_util.py", "entorno.py"):
    p = RAIZ / "scripts" / nombre
    if not p.exists():
        falla(f"falta scripts/{nombre}: no se borra nada de scripts/")

# --- Historial -------------------------------------------------------------
# El trabajo en equipo se evalua tambien por como se construyo la entrega.
git = RAIZ / ".git"
if git.is_dir():
    import subprocess

    r = subprocess.run(["git", "-C", str(RAIZ), "rev-list", "--count", "HEAD"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip().isdigit():
        n = int(r.stdout.strip())
        sugiere(n >= 5, f"solo {n} commit(s). Se espera trabajo repartido a lo "
                        "largo de la practica, no una entrega de golpe")

# ---------------------------------------------------------------------------

print()
if fallos:
    print(f"La entrega NO esta completa: {fallos} cosa(s) por resolver"
          + (f", {avisos} aviso(s)" if avisos else "") + ".")
    sys.exit(1)

if avisos:
    print(f"Entrega completa, con {avisos} aviso(s) que conviene mirar.")
else:
    print("Entrega completa.")
print("El rigor de la justificacion se valora en la defensa, y eso no lo "
      "comprueba ningun script.")
sys.exit(0)
