# Montar la práctica en casa, desde cero, con PyCharm

Guía de clics. Casi todo se hace con la interfaz; solo hay **cuatro líneas**
que se escriben, y van en la pestaña Terminal que PyCharm trae integrada abajo.

---

## El plan, en seis líneas

1. Montas PyCharm y el proyecto en tu PC.
2. Pruebas que todo funciona con las **12 muestras** que ya vienen en el repositorio.
3. Vas **una vez** a NIMROD: te traes el conjunto de 200 y sacas las tres filas
   de rendimiento (eso no se puede medir en casa).
4. En casa haces las cinco condiciones degradadas sobre los 200.
5. Juntas los dos CSV con un script.
6. Escribes el informe.

> **Lo único que no se puede hacer en casa** son el RTF, el tiempo de carga y la
> VRAM: el enunciado obliga a medirlos en NIMROD y `comprobar_entrega.py` lo
> verifica. El WER y el CER sí, donde quieras.

---

# PARTE 1 — Instalar las tres cosas base

### 1.1 Python

Descarga **Python 3.12** de [python.org/downloads](https://www.python.org/downloads/).

En el instalador de Windows, marca la casilla **«Add python.exe to PATH»** antes
de darle a Install. Es la casilla que todo el mundo se salta y luego da
problemas.

### 1.2 PyCharm

Descarga PyCharm de [jetbrains.com/pycharm/download](https://www.jetbrains.com/pycharm/download/).

La versión gratuita te vale: **el soporte de cuadernos Jupyter está incluido en
las funciones gratuitas**. Si además quieres las de pago, con el correo de la
UHU tienes licencia de estudiante gratis en
[jetbrains.com/community/education](https://www.jetbrains.com/community/education/).

### 1.3 ffmpeg

Hace falta para generar dos de las seis condiciones acústicas (`telefono` y
`codec`), y no se instala con pip.

Abre **PowerShell** (botón de Windows, escribe «PowerShell») y pega:

```
winget install Gyan.FFmpeg
```

Cuando termine, **cierra PowerShell**. Si luego PyCharm dice que no encuentra
ffmpeg, reinicia el ordenador: es que el PATH no se ha refrescado.

---

# PARTE 2 — Traer el proyecto a PyCharm

### 2.1 Conectar tu cuenta de GitHub

1. Abre PyCharm.
2. `File` → `Settings` (o `Ctrl+Alt+S`).
3. En el panel izquierdo: `Version Control` → `GitHub`.
4. Botón `+` → **`Log In via GitHub...`**. Se abre el navegador, aceptas, y
   PyCharm queda conectado.

Así te ahorras el *token* a mano: PyCharm lo gestiona él.

### 2.2 Clonar el repositorio

1. En la pantalla de bienvenida: **`Clone Repository`**.
   (Si ya tienes un proyecto abierto: `File` → `New` → `Project from Version Control`).
2. Pega la URL de **tu** repositorio de GitHub Classroom — la del botón verde
   `Code` de tu repo, no la de la plantilla.
3. Elige dónde guardarlo y dale a `Clone`.

PyCharm abre el proyecto. Vas a ver `p2_plantilla.ipynb`, `scripts/`,
`eval_local/` y las guías.

### 2.3 Meter los ficheros que te he preparado

Descomprime el ZIP que te pasé y **arrastra estos ficheros a la ventana de
PyCharm**, encima del nombre del proyecto. Cuando pregunte, dile *Overwrite*:

- `p2_plantilla.ipynb` (con el tercer backend ya metido)
- `transcribe.py`
- `comprobar_gpu.py`
- `fusionar_resultados.py`
- `informe/informe.md` e `informe/ficha_modelos.md`
- las guías `.md`

---

# PARTE 3 — El entorno y los paquetes

### 3.1 Crear el entorno virtual

Al abrir el proyecto, PyCharm suele detectar el `requirements.txt` y ofrecer
crear un entorno: **dile que sí** y te saltas este paso.

Si no lo ofrece:

1. `Ctrl+Alt+S` → `Project: <nombre>` → `Python Interpreter`.
2. `Add Interpreter` → `Add Local Interpreter`.
3. Tipo **`Virtualenv`**, base: el Python 3.12 que instalaste. `OK`.

Abajo a la derecha tiene que poner algo como `Python 3.12 (.venv)`.

### 3.2 Instalar los paquetes

Aquí van las cuatro líneas. Abre la pestaña **`Terminal`** abajo del todo de
PyCharm (o `Alt+F12`). Fíjate en que el prompt empiece por `(.venv)`: eso
significa que estás instalando dentro del entorno del proyecto.

Pega **una por una**, esperando a que acabe cada una:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```
pip install -r requirements.txt
```

```
pip install transformers
```

```
pip install vosk
```

> ⚠️ **La primera línea es la importante.** El README de la práctica dice
> `cu121`, y **esa versión no funciona con tu RTX 5060**: te saldría
> `no kernel image is available`. Si `cu128` te diera error de «no encuentro el
> paquete», entra en
> [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/),
> usa el selector (Windows / Pip / Python / la CUDA más alta que aparezca) y
> copia la línea que te dé.

La primera línea descarga unos 3 GB. Tarda.

---

# PARTE 4 — Comprobar que todo está bien (el botón verde)

1. En el panel de la izquierda, doble clic en **`comprobar_gpu.py`**.
2. Clic en el **triángulo verde ▶** de arriba a la derecha.

Lee la salida. Lo que tiene que salir:

```
[ OK ] GPU: NVIDIA GeForce RTX 5060
[info ] capacidad de computo: sm_120
[info ] es Blackwell: la misma familia que la GPU de NIMROD.
[ OK ] la GPU calcula de verdad (hay kernels para esta arquitectura)
[ OK ] CTranslate2 ve 1 GPU(s): faster-whisper ira por GPU
[info ] dispositivo() devuelve: (cuda, float16)
```

**Si sale `[FALLO] la GPU NO calcula`**, tu PyTorch es de una CUDA anterior a tu
tarjeta: repite la primera línea de la parte 3.2 con un número más alto.

**Si sale que CTranslate2 no ve la GPU**, en Windows suele faltar cuDNN. Pega en
el Terminal:

```
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Luego ejecuta también **`comprobar_entorno.py`** con el mismo botón verde: es el
del profesor y mira las dependencias.

---

# PARTE 5 — Primera prueba, con las 12 muestras

Todavía no tienes los 200 enunciados, pero puedes probar que el pipeline entero
funciona con las muestras que vienen en el repositorio.

1. Doble clic en **`p2_plantilla.ipynb`**. Se abre como cuaderno.
2. En la celda 2, cambia `EQUIPO = "equipo-XX"` por el de verdad.
3. Ejecuta las celdas **una a una** con `Mayús+Enter`, de arriba abajo.

Puntos de control:

| Celda | Qué tiene que salir |
|---|---|
| La primera | `dispositivo: cuda / float16` y un aviso de que usa las 12 muestras |
| 1.1 (normalizador) | `23:15` convertido a «veintitrés quince» |
| 1.2 | `12 enunciados` |
| 1.3 (baseline) | Texto transcrito y un WER |
| Descarga de Vosk | Baja 1,4 GB. Tarda; déjala correr |
| **Celda de humo** | **Los tres backends imprimen texto** |

**La celda de humo es la más importante de todas.** Si los tres backends
escriben una frase, ya tienes la práctica montada. Si alguno dice `FALLO:`,
arréglalo ahora: con 12 enunciados el error tarda diez segundos en aparecer, con
200 tardaría un cuarto de hora.

---

# PARTE 6 — Traerte el conjunto de NIMROD

Esto es lo que te permite medir el WER de verdad desde casa. Son unos 60 MB.

**Allí**, en un terminal de JupyterLab:

```
cd ~/equipo/p2-asr
tar czf eval_set.tar.gz -C "$(dirname $P2_EVAL_SET)" "$(basename $P2_EVAL_SET)"
```

Descárgalo con clic derecho → `Download` en el panel de ficheros.

**En casa:** descomprímelo y deja la carpeta dentro del proyecto, con el nombre
**`eval_set`**. Tiene que quedar así:

```
tu-proyecto/
├── eval_set/
│   ├── manifest.tsv
│   └── audio/
└── p2_plantilla.ipynb
```

Ya está. **No hay que configurar ninguna variable de entorno**: el cuaderno
detecta solo la carpeta `eval_set/` y la usa. Para confirmarlo, vuelve a darle
al botón verde en `comprobar_gpu.py`; tiene que decir:

```
[ OK ] eval_set/ presente con 200 enunciados: el cuaderno lo usara solo
```

Si dice 12, la carpeta no está donde toca o se llama distinto.

> `eval_set/` está en el `.gitignore`: es material del profesor y no se sube.

---

# PARTE 7 — La misma sesión, pero en NIMROD

Ve a NIMROD con el código ya probado. Objetivo único: **tres filas** con
`entorno=nimrod`, `condicion=limpio` y las tres columnas de rendimiento.

1. En PyCharm, sube tus cambios: pestaña `Commit` (izquierda) → marcas los
   ficheros → mensaje → `Commit and Push`.
2. En NIMROD: `git pull` en el terminal de JupyterLab.
3. Ejecuta el cuaderno hasta el final de la **sesión 2** y para ahí.
4. Copia el resultado con otro nombre:

```
cp resultados/resultados.csv resultados/resultados_nimrod.csv
```

5. Haz `git add` + `git push` de ese fichero, y en casa `git pull` (en PyCharm:
   `Git` → `Update Project`).

Los detalles de la sesión están en `GUIA_NIMROD.md`.

---

# PARTE 8 — La pasada buena, en casa

1. Ejecuta la celda de degradación del cuaderno (sesión 3). Tarda unos minutos.
2. **Escucha el audio degradado** con la celda 3.2, antes de mirar ninguna cifra.
   Si a 5 dB tú no entiendes la frase, no esperes que el modelo la acierte — y
   eso va al informe.
3. Ejecuta la pasada de las condiciones. Orientación de tiempos en tu 5060:

| Modelo | Las 5 condiciones × 200 enunciados |
|---|---|
| large-v3-turbo (GPU) | 15–25 min |
| wav2vec2-xlsr-53-es (GPU) | 10–15 min |
| vosk-model-es-0.42 (**CPU**) | 40–70 min |

4. Guarda el resultado aparte. En el Terminal:

```
cp resultados/resultados.csv resultados/resultados_local.csv
```

5. Junta los dos:

```
python fusionar_resultados.py resultados/resultados_nimrod.csv resultados/resultados_local.csv
```

Tienen que salir **18 filas**.

6. Comprueba la entrega: abre `comprobar_entrega.py` y dale al botón verde.

---

# PARTE 9 — Para tu compañera, que va con Mac

Lo primero, lo que hay que saber antes de repartir trabajo: **un Mac no tiene
GPU utilizable para esta práctica.** CTranslate2, que es el motor de
faster-whisper, **no tiene backend Metal**: Whisper va a correr por CPU aunque
el Mac sea un M3. Lo dice el propio README del profesor.

En la práctica eso significa que una pasada completa de 200 enunciados × 6
condiciones en su Mac se iría a horas. **No es el sitio donde ejecutar las
medidas.**

### 9.1 Lo que sí monta igual

Las partes 1, 2 y 5 son idénticas, salvo tres cambios:

**ffmpeg y Homebrew.** En la Terminal del Mac (Launchpad → Otras → Terminal):

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

```
brew install ffmpeg
```

**PyTorch normal, sin índice.** En el Mac no hay CUDA, así que la línea es
simplemente:

```
pip install torch
```

**Vosk con versión fijada.** Esto es importante y no es evidente: la última
versión de Vosk (0.3.45) **dejó de publicar instalador para Mac**. La anterior
sí lo tiene, y es universal (sirve para Intel y para Apple Silicon):

```
pip install vosk==0.3.44
```

Si pone `pip install vosk` a secas, le va a fallar.

Que ejecute **`comprobar_gpu.py`** igual que tú: el script detecta que es un Mac
y le dice qué esperar (si tiene MPS, que Whisper irá por CPU, y lo de Vosk).

### 9.2 Qué puede hacer ella, y hacerlo bien

Todo esto va perfecto en un Mac y es la mitad del trabajo de la práctica:

- **Las pruebas con los 12 enunciados** de `eval_local/`, para depurar los
  backends.
- **Generar el audio degradado.** El script es determinista: el audio que sale
  en su Mac es idéntico byte a byte al tuyo. Puede generarlo y escucharlo.
- **El análisis de errores**, que es la sección que más puntúa. Coger los
  veinte peores enunciados y clasificar los fallos: números, nombres propios,
  siglas, alucinaciones, y sobre todo contar cuántos **no son errores de
  reconocimiento** (lo de «sicología» / «psicología»).
- **Las fichas de los modelos:** licencias exactas, fechas del último release,
  actividad de los repositorios. Son dos de los seis criterios.
- **Redactar el informe.**
- **El round-trip con P1**, que es la parte bonificable.

### 9.3 Lo que NO debe hacer

**No debe medir el WER de la entrega con un backend distinto al tuyo.** El
README deja usar whisper.cpp o MLX en Mac, pero si ella mide el WER con
whisper.cpp y tú el RTF en NIMROD con faster-whisper, estáis comparando dos
implementaciones distintas del mismo modelo: `comprobar_entrega.py` lo detecta y
**rechaza la entrega**.

Regla simple: **un modelo, un backend, en todas sus filas.** Las cifras que
vayan al CSV salen de tu 5060 o de NIMROD.

---

# Reparto de trabajo sugerido

El informe tiene una sección de «quién hizo qué», así que conviene que esté
repartido de verdad.

| | Tú (RTX 5060) | Ella (Mac) |
|---|---|---|
| Montar entorno | Sí | Sí |
| Sesión en NIMROD | Sí | — |
| Pasadas de 200 enunciados | Sí | — |
| Generar y escuchar degradaciones | — | Sí |
| Análisis de errores (§6) | — | Sí |
| Fichas: licencias y mantenimiento (§2) | — | Sí |
| Protocolo de medida (§3) | Sí | — |
| Tabla y curva (§4, §5) | Sí | Sí |
| Redacción del informe | Sí | Sí |
| Round-trip con P1 (§8) | — | Sí |

---

# Si algo falla

| Lo que ves | Qué hacer |
|---|---|
| `no kernel image is available` | PyTorch de una CUDA vieja. Repite la parte 3.2 con `cu128` o superior |
| `CUBLAS_STATUS_NOT_SUPPORTED` | Estás usando `int8` en Blackwell. Tiene que ser `float16` |
| `Could not locate cudnn_ops64_9.dll` | `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` |
| `ffmpeg: command not found` | Reinicia PyCharm. Si sigue, reinicia el ordenador |
| El cuaderno dice `12 enunciados` | La carpeta no se llama exactamente `eval_set` o no está en la raíz del proyecto |
| `CUDA out of memory` | Dos modelos a la vez. Reinicia el kernel del cuaderno y vuelve a ejecutar |
| En Mac: `No matching distribution found for vosk` | `pip install vosk==0.3.44` |
| PyCharm no encuentra los paquetes | Mira abajo a la derecha: tiene que poner `(.venv)`. Si no, parte 3.1 |

Ante cualquier duda, el primer paso es siempre el mismo: abrir
`comprobar_gpu.py` y darle al botón verde.
