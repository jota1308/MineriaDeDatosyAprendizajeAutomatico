# TP Integrador — Minería de Datos y Aprendizaje Automático

Dataset asignado: **Nursery** (UCI, 12.960 solicitudes de ingreso a guarderías, 8 atributos categóricos).

## Tareas asignadas

1. **EDA** del dataset.
   - Subconjunto binario: todas las muestras de una clase mayoritaria + todas las de `very_recom`.
2. **Balanceo** de las clases seleccionadas con una técnica **híbrida** (sobre + submuestreo).
3. **Dos técnicas de clasificación**: 5 ejecuciones con distintos hiperparámetros cada una,
   medidas con cross-validation; promediar y comparar las métricas.
4. **Stacking** usando los 10 modelos del punto 3 como modelos débiles: 5 ejecuciones con CV,
   promediar y comparar contra el punto 3.

Entrega: código Python + informe PDF. Coloquio desde el 13/11, fecha límite 11/12 – 14/12.

## Progreso

- [ ] 1. EDA y armado del problema binario
- [ ] 2. Balanceo híbrido
- [ ] 3. Dos clasificadores × 5 configuraciones
- [ ] 4. Stacking × 5 configuraciones
- [ ] Informe PDF

## Estructura

```
data/raw/          dataset original (Datos.txt, Descripcion.txt, Tareas_a_realizar.txt)
notebooks/         notebook principal del TP
src/datos.py       carga del dataset con nombres de columnas
src/AA_utils.py    funciones de la cátedra (matriz de confusión, curvas ROC/PR, etc.)
resultados/        tablas y gráficos exportados para el informe
informe/           enunciado e informe final
```

## Cómo correrlo

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/TP_Nursery.ipynb
```
