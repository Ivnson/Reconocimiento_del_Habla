# Guía de ejecución — qué hacer en NIMROD y en qué orden

Documento de trabajo del equipo, no entregable. Resume lo que hay que hacer
cuando tengamos sesión en NIMROD, para no perder tiempo de laboratorio
decidiendo cosas allí.

**Estado actual:** el código está completo y los tres backends implementados.
Lo único que falta son las cifras, y esas solo salen en NIMROD, porque el
conjunto de evaluación real (200 enunciados) está montado allí y el RTF, el
tiempo de carga y la VRAM se miden obligatoriamente en ese hardware.

---

## Paso 0 — Llevar el código a NIMROD

El material que hay en el workspace es la plantilla del profesor, sin nuestros
cambios. Hay dos vías:

**Vía repositorio (preferible, porque además deja historial de commits, que se
evalúa):**

```bash
# en el portátil, dentro del repositorio
git add p2_plantilla.ipynb transcribe.py informe/ GUIA_NIMROD.md
git commit -m "Tercer backend (Vosk) y fichas de candidato"
git push

# en el terminal de JupyterLab de NIMROD
cd ~/equipo/p2-asr
git pull        # si el workspace es un clon del repo
```

**Vía panel de ficheros:** arrastrar `p2_plantilla.ipynb`, `transcribe.py` e
`informe/ficha_modelos.md` al panel de JupyterLab, sobre `equipo/p2-asr`.

Después, siempre:

```bash
cd ~/equipo/p2-asr
python3 comprobar_entorno.py      # ocho comprobaciones, ningún FALLO
```

---

## Paso 1 — El modelo de Vosk (hacerlo lo primero, mientras se habla de otra cosa)

Vosk no está en la caché compartida: hay que traerlo. Son 1,4 GB, una sola vez.
La celda del cuaderno lo hace sola, pero conviene lanzarlo desde el terminal al
principio de la sesión para que descargue mientras se trabaja en otra cosa:

```bash
cd ~/equipo/p2-asr
pip install vosk
python3 -c "
import urllib.request, zipfile, pathlib
d = pathlib.Path('modelos'); d.mkdir(exist_ok=True)
z = d / 'vosk-model-es-0.42.zip'
urllib.request.urlretrieve('https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip', z)
zipfile.ZipFile(z).extractall(d); z.unlink()
print('listo')
"
```

**Si NIMROD no tiene salida a internet**, descargar el zip en el portátil y
subirlo por el panel de ficheros a `~/equipo/p2-asr/modelos/`, descomprimido.
`modelos/` está en el `.gitignore`, así que no se sube al repositorio.

**Plan B si Vosk no arranca:** cambiar la tercera línea de `MODELOS` por
`("facebook/mms-1b-all", "transformers", "float32", "CC-BY-NC-4.0")`, que sí
está pre-descargado. Sabiendo que entonces **la comparativa pierde la tercera
familia** (MMS es CTC, igual que wav2vec2) y hay que decirlo en el informe.

---

## Paso 2 — Sesión 1: baseline

Celdas 1 a 1.4 del cuaderno, en orden. Lo único que hay que tocar antes:

```python
EQUIPO = "equipo-XX"     # <- poner el nuestro de verdad, celda 2
```

Qué debe salir:

- `conjunto de evaluación: /data/datasets/material-p2/eval_set` (o la ruta que
  traiga `P2_EVAL_SET`). **Si pone `eval_local`, la variable no está definida y
  estaríamos midiendo sobre 12 enunciados en vez de 200.** Reiniciar el
  workspace.
- `dispositivo: cuda / float16`.
- `200 enunciados, 32.5 minutos de audio`.
- La pasada del baseline con `small` ronda los dos minutos si no hay más
  equipos en la GPU.

**Apuntar en la bitácora**, que luego va al §6 del informe: WER y CER del
baseline, y los diez peores enunciados con una nota de qué les pasa. Esto se
escribe mucho mejor ahora que tres semanas después.

---

## Paso 3 — Sesión 2: los tres modelos en limpio

Orden de celdas: la de descarga de Vosk → la de los backends → **la celda de
humo** → `MODELOS` → la pasada completa.

**La celda de humo es la que ahorra la sesión.** Transcribe un solo enunciado
con cada uno de los tres backends. Si algo está mal, falla en diez segundos y
no a los quince minutos. Los tres deben imprimir texto; si alguno imprime
`FALLO:`, se arregla antes de seguir.

Tiempos aproximados de la pasada completa (200 enunciados, 32,5 min de audio):

| Modelo | Dónde corre | Tiempo esperado |
|---|---|---|
| large-v3-turbo | GPU | 2–4 min |
| wav2vec2-xlsr-53-es | GPU | 2–3 min |
| vosk-model-es-0.42 | **CPU** | 10–20 min |

Vosk es el que marca el reloj, porque no usa la GPU. Lanzar esa celda y
aprovechar para ir redactando el §3 del informe.

Al terminar, `resultados/resultados.csv` debe tener **3 filas** con
`entorno=nimrod`, `condicion=limpio` y las tres columnas de rendimiento
rellenas.

### El punto que hay que mirar con cuidado: la VRAM de Vosk

Vosk corre en CPU: **no reserva VRAM**. Lo que `MedidorVRAM` lea en su fila
será la memoria que el proceso del kernel siga teniendo reservada por el
contexto CUDA de los dos modelos anteriores, no la de Vosk.

Qué hacer: **dejar el número que salga** (el CSV lo exige no vacío) y decirlo
explícitamente en la ficha y en el §3 del informe. Es una limitación real del
protocolo de medida, no un error nuestro, y explicarla bien vale más que
cualquier cifra.

Conviene además medir la RAM de Vosk, que sí es su coste real, para poder
citarla en la ficha:

```python
import os, resource
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "MB de RAM pico del proceso")
```

Y preguntar al profesor en la sesión cómo quiere que se refleje: es justo el
tipo de detalle que puede salir en la defensa.

---

## Paso 4 — Sesión 3: las seis condiciones

1. Celda de degradación. Sobre 200 ficheros tarda unos minutos y es
   determinista. Genera `eval_degradado/` con cinco subcarpetas.
2. **Escuchar el audio antes de medir.** La celda 3.2 reproduce el mismo
   enunciado en las cinco condiciones. A 5 dB de SNR, si nosotros no entendemos
   la frase, no hay que esperar que el modelo la acierte — y eso es parte del
   análisis, no una excusa.
3. La pasada de las 18 combinaciones. Con Vosk en CPU, contar **40–70 minutos**
   en total. Lanzarla y no tocar el kernel.

Al terminar: `resultados.csv` con **18 filas**, y la curva de degradación.

Predicciones que conviene contrastar con lo que salga (si fallan, también es
material para el informe):

- `telefono` debería hacer más daño a wav2vec2 que a Whisper: al cortar por
  encima de 4 kHz se pierden las fricativas, y un modelo sin modelo de lenguaje
  no tiene con qué recuperarlas.
- `snr5` debería ser donde Whisper empiece a **alucinar**, no solo a fallar:
  frases inventadas, repeticiones, texto al final que no está en el audio. Eso
  puede disparar el WER por encima de 1 por inserciones.
- Vosk debería degradarse de forma más plana: peor en limpio, pero con menos
  caída, porque el n-grama restringe lo que puede salir.

---

## Paso 5 — Cierre

```bash
python3 comprobar_entrega.py
```

Y antes de entregar, en el cuaderno: **Kernel → Restart and Run All**. Se
entrega con todas las celdas ejecutadas y con salida.

Lista de cierre:

- [ ] `EQUIPO` puesto de verdad, no `equipo-XX`
- [ ] `resultados/resultados.csv` con 18 filas
- [ ] Una fila `nimrod` + `limpio` por modelo con RTF, t_carga y VRAM
- [ ] `informe/ficha_modelos.md` sin ningún `⟨PENDIENTE⟩`
- [ ] `transcribe.py` apuntando al modelo que recomendemos en el §7
- [ ] `informe/informe.pdf` exportado
- [ ] Commits repartidos, no uno solo al final

---

## Problemas que ya sabemos que van a aparecer

| Síntoma | Qué es | Qué hacer |
|---|---|---|
| El cuaderno dice `eval_local` | `P2_EVAL_SET` no está definida | Reiniciar el workspace. No seguir: serían 12 enunciados, no 200 |
| `CUDA error` al cargar | `int8` en GPU Blackwell | `float16` siempre. `dispositivo()` ya lo resuelve |
| RTF por encima de 1 | No se descartó el calentamiento | `calentar()` antes de cronometrar |
| VRAM = 0 MB | Se midió con `torch.cuda.max_memory_allocated()` | `MedidorVRAM`, que pregunta al driver por NVML |
| `Read-only file system` | HF intenta escribir en la caché compartida | Nada: `silenciar_avisos_cache()` ya está en la celda 1 |
| `CUDA out of memory` | Dos modelos vivos a la vez | `del modelo` + `torch.cuda.empty_cache()` (ya está en el bucle) |
| Vosk no encuentra el modelo | No se descargó, o está en otra carpeta | Paso 1, o `export VOSK_MODEL_PATH=/ruta/al/modelo` |
| Vosk tarda muchísimo | Es CPU, y el contenedor tiene pocos núcleos | Normal. `cpus_limite()` dice cuántos hay |
