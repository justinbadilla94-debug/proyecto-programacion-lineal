# Programación lineal: método Simplex y método gráfico

Herramienta en Python con interfaz en Streamlit para resolver problemas de programación lineal
con dos variables de decisión y tres restricciones (signos ≤, ≥ e =), mediante:

- **Método Simplex** con el método de la Gran M (muestra el sistema transformado, cada iteración, las razones, el pivote y las operaciones de Gauss-Jordan).
- **Método gráfico** (rectas de las restricciones, región factible, vértices y evaluación de la función objetivo).

Detecta problemas infactibles y no acotados, y compara el resultado de ambos métodos.

## Archivos

| Archivo | Contenido |
|---|---|
| `app.py` | Interfaz en Streamlit |
| `simplex.py` | Clase `MetodoSimplex` |
| `grafico.py` | Clase `MetodoGrafico` |
| `requirements.txt` | Bibliotecas necesarias |

## Ejecutar localmente

```
pip install -r requirements.txt
streamlit run app.py
```

Curso IF0017 Métodos Numéricos y Análisis Computacional, UCR Sede de Occidente, II ciclo 2026.
Anthonny López Martínez y Justin Badilla Ramírez.
