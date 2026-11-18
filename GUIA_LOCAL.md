# Guía paso a paso — hacer la P2 en local con una RTX 5060

---

## Antes de empezar: qué se puede y qué no

Hay que ser honesto con esto, porque condiciona todo el plan.

**Sí se puede en local:** el WER y el CER. El enunciado lo dice literalmente —
«El RTF, el tiempo de carga y la VRAM se miden **en NIMROD**. El WER y el CER,
donde queráis». Con la 5060 vas a ir sobrado para todo eso.

**No se puede en local, y no hay forma de saltárselo:**

1. **El RTF, el tiempo de carga y la VRAM.** Tienen que salir de NIMROD. No es
   un capricho: un RTF medido en tu 5060 y otro medido en la RTX PRO 5000 de
   NIMROD no se pueden poner en la misma tabla de clase. `comprobar_entrega.py`
   exige una fila `entorno=nimrod` + `condicion=limpio` por modelo con esas tres
   columnas rellenas, y la rechaza si no está.
2. **El conjunto de 200 enunciados** está montado en NIMROD, en
   `/data/datasets/material-p2/eval_set`. En tu PC solo tienes los 12 de
   `eval_local/`, que explícitamente **no son el conjunto de evaluación**.

**El plan que sale de ahí**, y que es el que desarrolla esta guía:

| Dónde | Qué | Cuánto tarda |
|---|---|---|
| **Casa (ahora)** | Montar el entorno, depurar los tres backends con los 12 enunciados de muestra | Una tarde |
| **NIMROD (sesión corta)** | Copiarte el `eval_set` + la pasada de `limpio` con rendimiento | 45–60 min |
| **Casa** | Las cinco condiciones degradadas sobre los 200, análisis, informe | Lo que haga falta |
| **Casa** | Fusionar los dos CSV y cerrar la entrega | 5 min |

Con esto, en NIMROD solo gastas una sesión corta y bien aprovechada, y todo el
trabajo lento y de análisis lo haces en casa con tu propia GPU.

---

## Lo primero que tienes que saber de tu GPU

**La RTX 5060 es Blackwell, arquitectura sm_120 — exactamente la misma familia
que la GPU de NIMROD.** Esto tiene tres consecuencias directas:

1. **`float16` siempre, `int8` nunca** con faster-whisper. El mismo fallo que
   avisa el enunciado para NIMROD te va a pasar en casa: CTranslate2 revienta
   con `int8` en Blackwell. Es un fallo conocido y reportado en las RTX 50.
2. **La línea del README está obsoleta y te va a fallar.** El README dice
   `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
   **CUDA 12.1 no soporta sm_120.** Si lo instalas así te saldrá
   `no kernel image is available for execution on the device`. Necesitas CUDA
   **12.8 o superior**.
3. **Comprueba cuánta VRAM tienes**, porque decide qué modelos puedes meter. La
   5060 normal lleva 8 GB; la 5060 Ti puede llevar 8 o 16. Lo ves con
   `nvidia-smi` (paso 1).

Lo bueno: como tu GPU y la de NIMROD son la misma arquitectura, la
configuración que te funcione en casa te funcionará allí sin cambiar nada. Eso
es más suerte de la que parece.

---

## Paso 1 — Comprobar la GPU y el driver

Abre una terminal (PowerShell en Windows) y ejecuta:

```bash
nvidia-smi
```

Apunta dos cosas de la salida:

- **`CUDA Version`** arriba a la derecha. Tiene que ser **12.8 o superior**. Si
  es menor, actualiza el driver de NVIDIA antes de seguir: es lo primero que hay
  que arreglar y lo más fácil de pasar por alto.
- **La memoria total**, en la columna `Memory-Usage` (algo como `/ 8192MiB`).
  Ese número decide el paso 5.

---

## Paso 2 — Clonar y preparar el entorno

```bash
git clone https://github.com/UHU-ETSI-PH-2026/<tu-repositorio>.git
cd <tu-repositorio>

python3 -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / WSL:
source .venv/bin/activate
```

> El repositorio es privado: git te pedirá un *personal access token*, no la
> contraseña. Se saca en GitHub → Settings → Developer settings → Personal
> access tokens → Tokens (classic), con el ámbito `repo`.

### ffmpeg

Hace falta para las condiciones `telefono` y `codec`, y no se instala con pip.

```bash
# Windows:
winget install Gyan.FFmpeg
# Linux / WSL:
sudo apt install ffmpeg
```

Cierra y vuelve a abrir la terminal, y comprueba que está en el PATH:

```bash
ffmpeg -version
```

Si no lo encuentra, el PATH no se ha refrescado: reinicia la terminal. En
Windows a veces hace falta cerrar sesión.

---

## Paso 3 — Instalar, en este orden

El orden importa. PyTorch primero, porque es el que arrastra las bibliotecas
de CUDA.

```bash
# 1. PyTorch con CUDA compatible con Blackwell. Entra en
#    https://pytorch.org/get-started/locally/ y usa el selector: elige tu
#    sistema, Pip, Python y la CUDA mas alta que ofrezca (12.8 o superior).
#    Te dara una linea parecida a esta:
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. El resto de la practica
pip install -r requirements.txt

# 3. El backend CTC (lo trae comentado requirements.txt a proposito)
pip install transformers

# 4. El tercer backend
pip install vosk
```

**No uses la línea `cu121` del README.** Es lo único del material que no vale
para tu GPU.

---

## Paso 4 — Verificar que Blackwell funciona DE VERDAD

Este paso parece tonto y es el que más tiempo ahorra. Un `torch.cuda.is_available()`
que devuelve `True` **no significa que vaya a funcionar**: en una GPU sin kernels
compilados, el fallo no aparece hasta que intentas calcular algo.

```bash
python3 -c "
import torch
print('torch     :', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('GPU       :', torch.cuda.get_device_name(0))
print('capacidad :', torch.cuda.get_device_capability(0), '  <- tiene que ser (12, 0)')
x = torch.randn(2000, 2000, device='cuda')
print('calculo   :', float((x @ x).sum()), '  <- si llega aqui, los kernels sm_120 estan)')
"
```

Si la última línea revienta con `no kernel image is available`, tu PyTorch está
compilado para una CUDA anterior: vuelve al paso 3 con un índice más alto.

Ahora CTranslate2, que es el motor de faster-whisper y va por su cuenta:

```bash
python3 -c "
import ctranslate2
print('ctranslate2:', ctranslate2.__version__, '| GPUs vistas:', ctranslate2.get_cuda_device_count())
from scripts.asr_util import dispositivo
print('dispositivo():', dispositivo(), '  <- tiene que decir (cuda, float16)')
"
```

Y por último, la comprobación del profesor:

```bash
python3 comprobar_entorno.py
```

Ocho comprobaciones. Los `AVISO` no bloquean; los `FALLO`, sí. Es normal que
avise de que no encuentra el conjunto de evaluación: eso lo arreglamos en el
paso 6.

---

## Paso 5 — Elegir los modelos (aquí es donde ganas con la 5060)

Tienes razón en que puedes subir de nivel, pero con dos restricciones que
conviene tener claras antes de elegir:

- **El modelo que uses en casa tiene que ser el mismo que midas en NIMROD.**
  Si mides el WER con `large-v3` en casa y el RTF con `large-v3-turbo` allí,
  `comprobar_entrega.py` lo rechaza y además no sería comparable. Elige ahora y
  no cambies.
- **Más grande no es automáticamente mejor en español.** Un modelo mediano
  ajustado específicamente a español suele batir a uno enorme multilingüe.

### Lo que te recomiendo

| Familia | Cambio | Por qué |
|---|---|---|
| encoder-decoder | `large-v3-turbo` → **`large-v3`** | Mejora gratis: `turbo` es la variante con el decodificador podado. `large-v3` ocupa ~3,1 GB en float16, te cabe en 8 GB, y **ya está en la caché de NIMROD**, así que no tienes que descargarlo allí. |
| CTC | **dejar `wav2vec2-large-xlsr-53-spanish`** | Está ajustado a español. `mms-1b-all` es tres veces más grande pero genérico multilingüe, y para español concreto lo normal es que pierda. Además ocuparía ~4 GB en float32, que en 8 GB va justo. |
| híbrido | **dejar `vosk-model-es-0.42`** | Es la tercera familia, pesa poco y funciona igual en los dos sitios. |

Para aplicarlo, en la celda `MODELOS` del cuaderno cambia solo la primera línea:

```python
("large-v3",  "faster-whisper", PRECISION, "MIT"),
```

### Si quieres ir a por nota

El candidato ambicioso es **NVIDIA Canary (NeMo)**, familia Conformer, que
encabeza el Open ASR Leaderboard. En casa tienes algo que no tienes en el
laboratorio: tiempo para pelearte con la instalación sin gastar sesión.

Pero antes de meterte, ten presente que **tendrías que instalarlo también en
NIMROD** para la fila de rendimiento, y NeMo no está en la caché de allí. Si
Canary te funciona en casa pero no en NIMROD, te quedas sin la fila obligatoria
y has perdido el modelo.

Mi consejo: cierra primero la práctica entera con los tres de arriba. Si te
sobra tiempo, pruebas Canary como cuarto candidato — y si falla, documentas el
error exacto y cuenta igual como candidato explorado en el informe (tarea 2.2).

---

## Paso 6 — Traerte el conjunto de evaluación de NIMROD

Esto es lo que te permite medir el WER sobre los 200 de verdad desde casa. Son
unos 60 MB. El material es CC-BY (FLEURS) y CC0 (VoxPopuli), así que copiarlo
no tiene problema; la atribución de FLEURS va en el informe, como pide el
enunciado.

**En NIMROD**, en un terminal de JupyterLab:

```bash
cd ~/equipo/p2-asr
echo $P2_EVAL_SET          # apunta el valor que salga
tar czf eval_set.tar.gz -C "$(dirname $P2_EVAL_SET)" "$(basename $P2_EVAL_SET)"
ls -lh eval_set.tar.gz
```

Descárgalo con el panel de ficheros de JupyterLab (clic derecho → Download).

**En casa**, dentro del repositorio:

```bash
tar xzf eval_set.tar.gz
mv <carpeta-que-salga> eval_set
ls eval_set/manifest.tsv       # tiene que existir
```

Y ahora apuntas la variable al conjunto antes de abrir Jupyter:

```bash
# Windows PowerShell:
$env:P2_EVAL_SET = "eval_set"
# Linux / WSL:
export P2_EVAL_SET=eval_set

jupyter lab
```

El cuaderno lee esa variable en la celda 1. **Comprueba que la primera celda
imprime `200 enunciados, 32.5 minutos`.** Si pone 12, la variable no ha llegado
al kernel: cierra Jupyter, vuelve a exportarla y ábrelo otra vez desde la misma
terminal.

> `eval_set/` ya está en el `.gitignore`: es material del profesor y no se sube.

---

## Paso 7 — La sesión en NIMROD (corta y con un objetivo)

Vas con el código ya depurado, así que esto es mecánico. Objetivo: **tres filas**
con `entorno=nimrod`, `condicion=limpio` y RTF, tiempo de carga y VRAM rellenos.

1. Sube tu código (`git push` en casa, `git pull` allí).
2. Descarga el modelo de Vosk allí — ver el paso 1 de `GUIA_NIMROD.md`.
3. Ejecuta el cuaderno hasta el final de la **sesión 2** (la pasada de `limpio`),
   y para ahí. No hace falta que hagas las degradaciones en NIMROD.
4. Renombra el CSV para que no te lo pise la pasada de casa:

```bash
cp resultados/resultados.csv resultados/resultados_nimrod.csv
```

5. Descárgatelo, o haz `git add resultados/resultados_nimrod.csv` y `git push`.

Lo que tiene que salir: `dispositivo: cuda / float16`, `200 enunciados`, y tres
líneas con WER, CER, RTF y VRAM.

---

## Paso 8 — Las cinco condiciones degradadas, en casa

```bash
python3 -I scripts/degradar.py --eval-set eval_set --salida eval_degradado
```

Sobre 200 ficheros tarda unos minutos. Es determinista: el audio que te sale en
casa es idéntico byte a byte al que saldría en NIMROD, así que las cifras son
igual de válidas.

Luego ejecuta la **sesión 3** del cuaderno. Tiempos orientativos en una 5060:

| Modelo | Las 5 condiciones × 200 enunciados |
|---|---|
| large-v3 (float16, GPU) | 20–35 min |
| wav2vec2-xlsr-53-es (GPU) | 10–15 min |
| vosk-model-es-0.42 (**CPU**) | 40–70 min |

Vosk es el que marca el reloj porque no usa la GPU. Lánzalo y vete a hacer otra
cosa.

Al terminar, renombra:

```bash
cp resultados/resultados.csv resultados/resultados_local.csv
```

**Escucha el audio degradado antes de mirar las cifras** (celda 3.2). Si a 5 dB
tú no entiendes la frase, no esperes que el modelo la acierte — y eso es parte
del análisis.

---

## Paso 9 — Fusionar los dos CSV

```bash
python3 fusionar_resultados.py resultados/resultados_nimrod.csv \
                               resultados/resultados_local.csv
```

El script deja una sola fila por (modelo, condición), hace ganar a las de NIMROD
para `limpio` —que son las que llevan el rendimiento válido— y **vacía
`rtf_medio`, `t_carga_s` y `vram_pico_mb` en las filas locales**, que es
exactamente lo que exige el enunciado. No estima ni copia ningún número.

Tienen que salir **18 filas**. Luego:

```bash
python3 comprobar_entrega.py
```

---

## Paso 10 — Cerrar

- [ ] `EQUIPO` puesto de verdad en la celda 2, no `equipo-XX`
- [ ] `transcribe.py` apuntando al modelo que recomiendes en el §7 del informe
- [ ] `informe/ficha_modelos.md` sin ningún `⟨PENDIENTE⟩`
- [ ] En el §3.6 del informe, **di explícitamente qué mediste en casa y qué en
      NIMROD**. El enunciado lo pide, y decirlo suma; ocultarlo y que se note,
      resta
- [ ] Kernel → **Restart and Run All** antes de entregar
- [ ] `informe/informe.pdf` exportado
- [ ] Commits repartidos a lo largo de la práctica

---

## Problemas típicos en Blackwell y en Windows

| Síntoma | Causa | Solución |
|---|---|---|
| `no kernel image is available for execution on the device` | PyTorch compilado para CUDA < 12.8; sm_120 no está en la rueda | Reinstalar torch desde el índice cu128 o superior (paso 3) |
| `CUBLAS_STATUS_NOT_SUPPORTED` al cargar con faster-whisper | `int8` en Blackwell | `float16` siempre. `dispositivo()` ya lo devuelve bien |
| `Could not locate cudnn_ops64_9.dll` (Windows) | CTranslate2 necesita las bibliotecas CUDA 12 y no las encuentra | `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` |
| `torch.cuda.is_available()` da `True` pero todo va lentísimo | Está cayendo a CPU en silencio | Mira `dispositivo()`: si dice `cpu/int8`, CTranslate2 no ve la GPU |
| `CUDA out of memory` | Dos modelos vivos a la vez, u 8 GB justos | `del modelo` + `torch.cuda.empty_cache()` (ya está en el bucle del cuaderno). Si persiste, baja de `large-v3` a `large-v3-turbo` |
| `ffmpeg: command not found` al degradar | No está en el PATH | Reinstalar y reiniciar la terminal (paso 2) |
| El cuaderno dice `12 enunciados` | `P2_EVAL_SET` no llegó al kernel | Exportar la variable y abrir Jupyter **desde esa misma terminal** (paso 6) |
| Vosk no encuentra el modelo | No está descomprimido en `modelos/` | `export VOSK_MODEL_PATH=/ruta/al/modelo`, o la celda de descarga |
| VRAM = 0 MB | Se midió con `torch.cuda.max_memory_allocated()` | `MedidorVRAM`, que pregunta al driver por NVML |

---

## Y una cosa sobre la nota

El enunciado lo dice al final del README y merece la pena tomárselo en serio:
*«Conseguir un WER bajo no da puntos por sí solo. Un WER alto, bien medido,
explicado y atribuido a una causa concreta vale más que un WER bajo sin
explicaciones»*.

Meter `large-v3` en vez de `large-v3-turbo` te bajará el WER un poco. Entender
**por qué** Vosk se degrada de forma más plana que Whisper a 5 dB de SNR, o por
qué la condición `telefono` castiga más al modelo CTC, te va a dar bastantes más
puntos — y es justo lo que preguntan en la defensa individual.
