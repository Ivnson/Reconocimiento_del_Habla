# Práctica 2 — Reconocimiento automático del habla

**Procesamiento del Habla · Grado en Ingeniería Informática · ETSI · Universidad de Huelva**

Este repositorio es el punto de partida de la Práctica 2. Contiene el cuaderno
de trabajo y las funciones de apoyo; el enunciado completo, con las tareas y la
rúbrica, está en el documento de la práctica publicado en Moodle.

El material funciona **igual en el servidor NIMROD y en vuestro propio
ordenador**. El código detecta dónde se ejecuta y ajusta rutas y dispositivo de
cómputo sin que tengáis que cambiar nada.

---

## Opción A — Trabajar en NIMROD (recomendada)

Es la opción por defecto: tiene GPU, los modelos ya están cacheados y el
conjunto de evaluación completo está montado.

1. Desde una de las workstations del LAB-104, entrad en `http://nimrod:7081`.
2. Crear workspace con la plantilla `user-gpu`, poniendo vuestro número de equipo.
3. En JupyterLab, abrid `equipo` → `p2-asr` → `p2_plantilla.ipynb`.

El material ya está en vuestro espacio de trabajo. Si necesitáis una copia
limpia porque habéis roto algo:

```python
import shutil
shutil.copytree('/data/datasets/material-p2/plantilla', '/data/equipo/p2-asr',
                dirs_exist_ok=True)
```

Ojo: `dirs_exist_ok=True` sobreescribe los ficheros con el mismo nombre. Si
queréis conservar vuestro trabajo, copiadlo antes a otro sitio.

Para entregar, copiad vuestro trabajo a este repositorio y haced *commit*
(ver «Qué se entrega», más abajo).

## Opción B — Trabajar en vuestro ordenador

Viable si tenéis máquina suficiente. Sin GPU todo funciona, solo que más
despacio: con 200 enunciados y seis condiciones acústicas, una pasada completa
puede irse a horas en CPU.

**Requisitos mínimos razonables:** 16 GB de RAM, 25 GB libres en disco y Python
3.10 o superior. Con GPU NVIDIA, mejor.

### 1. Clonar vuestro repositorio

La URL es la de **vuestro** repositorio, el que GitHub Classroom creó al
aceptar la tarea, no el de la plantilla. La encontraréis en el botón verde
**Code** de vuestro repositorio.

```bash
git clone https://github.com/UHU-ETSI-PH-2026/<vuestro-repositorio>.git
cd <vuestro-repositorio>
```

Los repositorios son **privados**, así que git os pedirá credenciales. La
contraseña de GitHub ya no vale: hace falta un *personal access token*.

1. En GitHub: Settings → Developer settings → Personal access tokens → Tokens
   (classic) → Generate new token, con el ámbito **`repo`**.
2. Copiadlo (solo se ve una vez) y pegadlo cuando git pida la contraseña.

Para no tener que repetirlo en cada operación:

```bash
git config --global credential.helper store        # Linux
git config --global credential.helper osxkeychain  # macOS
git config --global credential.helper manager      # Windows
```

### 2. Instalar

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python3 comprobar_entorno.py     # comprueba que todo está en su sitio
```

También hace falta **ffmpeg**, que no se instala con pip:
`sudo apt install ffmpeg` o `brew install ffmpeg`.

`comprobar_entorno.py` debe terminar con «Entorno listo» y sin ningún FALLO.
Si lo hace, ya podéis abrir el cuaderno:

```bash
jupyter lab
```

### 3. Si elegís un modelo de la familia CTC

`requirements.txt` trae faster-whisper, que es lo que usa la sesión 1. Los
modelos CTC (wav2vec2, MMS) necesitan además PyTorch, que ocupa un par de GB y
por eso no va en el núcleo:

```bash
pip install transformers torch
```

Con GPU NVIDIA, instalad primero la rueda de vuestra versión de CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Qué cambia respecto a NIMROD

El cuaderno **detecta dónde se ejecuta** y ajusta dos cosas sin que toquéis
nada:

- **El conjunto.** En NIMROD usa el conjunto común completo, que llega por la
  variable `P2_EVAL_SET`. En local usa `eval_local/`, doce enunciados de
  muestra. **Eso no es el conjunto de evaluación**: los números que entregáis
  salen de los 200 enunciados completos, en NIMROD.
- **El dispositivo.** `dispositivo()` devuelve `cuda`/`float16` donde hay GPU
  NVIDIA y `cpu`/`int8` donde no. No escribáis `"cuda"` a mano en ninguna
  celda: la GPU de NIMROD exige `float16` e `int8` falla, mientras que en CPU
  `float16` no existe. Son combinaciones incompatibles.

En macOS con chip Apple, CTranslate2 (el motor de faster-whisper) no tiene
backend Metal y usa la CPU. Para aprovechar la GPU del Mac la vía es whisper.cpp
o MLX; ambos son backends válidos y, si los usáis, debe quedar anotado en la
columna `backend`.

---

## Lo primero: 

1. **`float16`, nunca `int8`.** La GPU de NIMROD es Blackwell (sm_120) y
   faster-whisper con `compute_type="int8"` falla ahí.
2. **No forcéis descargas de modelos en NIMROD.** Los pesos están
   pre-descargados en la caché compartida, que es de sólo lectura. Ocho equipos
   bajando `large-v3` a la vez saturan la red.
3. **RTF, tiempo de carga y VRAM se miden en NIMROD, siempre.** El WER y el
   CER, donde queráis.

---

## Qué hay en el repositorio

| Ruta | Qué es |
|---|---|
| `p2_plantilla.ipynb` | Cuaderno de trabajo, con las tres sesiones. Por aquí se empieza |
| `scripts/normalize.py` | Normalizador canónico de texto. La única fuente de verdad del WER |
| `scripts/metrics.py` | WER, CER, RTF, medida de VRAM, esquema del CSV, `calentar()` |
| `scripts/asr_util.py` | `sin_ruido()`, límites reales del contenedor, carga del manifiesto |
| `scripts/entorno.py` | Verificación del entorno, compartida con `comprobar_entorno.py` |
| `scripts/degradar.py` | Genera las seis condiciones acústicas, de forma determinista |
| `eval_local/` | Muestra pequeña del conjunto, para trabajar sin NIMROD |
| `informe/informe.md` | Esqueleto del informe |
| `transcribe.py` | Vuestro entregable: `transcribe(ruta_audio) -> str` |
| `comprobar_entorno.py` | Verifica que las dependencias están instaladas |
| `comprobar_entrega.py` | Verifica que la entrega está completa |

Los ficheros de `scripts/` están comentados y explican **por qué** cada medida
se toma como se toma. Leedlos: de eso se pregunta en la defensa individual.

**`scripts/` no se modifica.** Si cambiáis algo de ahí, vuestros números dejan
de ser comparables con los del resto de la clase y la tabla agregada de la
defensa no os incluye.

---

## Qué se entrega

Se entrega **en este repositorio**, con *commits* a lo largo de la práctica (no
uno solo al final: el historial forma parte de la evaluación del trabajo en
equipo).

```
informe/informe.pdf        el informe, máximo 15 páginas sin anexos
p2_plantilla.ipynb         con todas las celdas ejecutadas
transcribe.py              con vuestro modelo elegido y la carga cacheada
resultados/resultados.csv  esquema fijo, mínimo 18 filas
```

Antes de entregar, ejecutad:

```bash
python3 comprobar_entrega.py
```

También se ejecuta solo en cada push: el resultado aparece en la pestaña
**Actions**, como un resumen de lo que falta. No bloquea nada — durante la
práctica es normal que la entrega esté incompleta.

### El CSV

Esquema fijo. No se amplía, no se reordena: la tabla agregada de la clase se
monta concatenando el CSV de todos los equipos.

```
equipo,modelo,backend,precision,condicion,entorno,wer,cer,rtf_medio,t_carga_s,vram_pico_mb,licencia
```

- `entorno` vale `nimrod` o `local`.
- Las filas `local` llevan WER y CER, y dejan **vacías** `rtf_medio`,
  `t_carga_s` y `vram_pico_mb`. **No las estiméis ni las copiéis** de otra fila.
- Debe haber al menos una fila `entorno=nimrod` + `condicion=limpio` por cada
  modelo, con las tres columnas de rendimiento rellenas.

### Qué NO se sube

El `.gitignore` ya excluye el audio degradado, los modelos descargados y las
cachés. **No los forcéis**: el audio degradado son cientos de ficheros que se
regeneran con un comando, y es determinista, así que a cualquiera que clone el
repositorio le saldrá exactamente el mismo.

La excepción es `eval_local/`, que sí viaja con el repositorio: son las pocas
muestras que permiten arrancar el cuaderno sin NIMROD.

---

## Tres errores típicos:

**Medir la VRAM con `torch.cuda.max_memory_allocated()`.** No ve la memoria de
CTranslate2, porque faster-whisper no usa el asignador de PyTorch. Devuelve
0 MB y parece que funciona. Usad `MedidorVRAM`, que pregunta al driver vía NVML
y sirve para cualquier backend.

**No descartar las pasadas de calentamiento.** La primera inferencia de un
proceso paga el contexto CUDA y la compilación de kernels. Medido en NIMROD con
faster-whisper `small`: **RTF 1,403 en la primera pasada frente a 0,053 en las
siguientes**, un factor de 26 que no tiene nada que ver con el modelo. Usad
`calentar()` antes de cronometrar.

**Mezclar backends entre filas.** Si medís el WER en vuestro Mac con
whisper.cpp y el RTF en NIMROD con faster-whisper, estáis comparando dos
implementaciones distintas del mismo modelo. `comprobar_entrega.py` os lo
rechaza.

---

## Y una cosa sobre la nota

Conseguir un WER bajo no da puntos por sí solo. Un WER alto, bien medido, explicado y
atribuido a una causa concreta vale más que un WER bajo sin explicaciones...
