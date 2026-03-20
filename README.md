# Numeric Calculus Project

Proyecto para detectar la silueta principal de una imagen y aproximar sus contornos superior e inferior con splines cubicos naturales.

## Requisitos

- Python 3.12 o compatible
- Dependencias de `requirements.txt`

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecucion

Desde la raiz del repositorio:

```powershell
python .\cubic_spline\main.py
```

Ese comando usa por defecto la imagen `img/G6.jpg` y guarda resultados en `outputs/`.

Cuando la imagen de entrada es `G6.jpg`, el programa tambien genera una segunda salida en `output_g6/` con un contorno superior refinado a partir de puntos de control especificos de esa imagen.

Tambien puedes indicar otra imagen o cambiar parametros:

```powershell
python .\cubic_spline\main.py --input .\img\G6.jpg --output-dir .\outputs --compare-scipy
```

```powershell
python .\cubic_spline\main.py --input .\img\G6.jpg --grabcut-iters 5 --grabcut-margin 0.05 --sample-step 4 --show
```

## Salidas

El programa crea:

- `outputs/plots/` con las figuras PNG
- `outputs/data/` con los puntos y curvas exportados a CSV
- `output_g6/plots/` y `output_g6/data/` cuando la imagen procesada es `G6.jpg`

## Parametros principales

- `--input`: ruta de imagen a procesar
- `--output-dir`: directorio de salida
- `--grabcut-iters`: iteraciones de GrabCut
- `--grabcut-margin`: margen relativo del rectangulo inicial
- `--sample-step`: separacion horizontal entre puntos muestreados
- `--compare-scipy`: compara el spline manual con SciPy
- `--show`: muestra las figuras ademas de guardarlas
