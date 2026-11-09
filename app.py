import numpy as np
import pandas as pd
import streamlit as st
from simplex import MetodoSimplex, TOL
from grafico import MetodoGrafico

st.set_page_config(page_title="Programacion lineal", layout="wide")
st.title("Programacion lineal")
st.write("Metodo Simplex (Gran M) y metodo grafico")

# ---- Datos de entrada (barra lateral) ----
st.sidebar.header("Datos del problema")
tipo = st.sidebar.selectbox("Tipo de problema", ["Maximizar", "Minimizar"])
c1 = st.sidebar.number_input("Coeficiente de x1 en Z", value=3.0)
c2 = st.sidebar.number_input("Coeficiente de x2 en Z", value=5.0)

por_defecto = [(1.0, 0.0, "<=", 4.0), (0.0, 2.0, "<=", 12.0), (3.0, 2.0, "<=", 18.0)]
restricciones = []
for i in range(3):
    st.sidebar.subheader(f"Restriccion {i + 1}")
    a = st.sidebar.number_input("Coeficiente de x1", value=por_defecto[i][0], key=f"a{i}")
    b = st.sidebar.number_input("Coeficiente de x2", value=por_defecto[i][1], key=f"b{i}")
    signo = st.sidebar.selectbox("Signo", ["<=", ">=", "="],
                                 index=["<=", ">=", "="].index(por_defecto[i][2]),
                                 key=f"s{i}")
    d = st.sidebar.number_input("Lado derecho", value=por_defecto[i][3], key=f"d{i}")
    restricciones.append((a, b, signo, d))

resolver = st.sidebar.button("Resolver problema", type="primary",
                             use_container_width=True)

# ---- Modelo matematico ----
st.subheader("Modelo matematico")
modelo = ("Max" if tipo == "Maximizar" else "Min") + f" Z = {c1:g}x1 + {c2:g}x2\n"
for a, b, signo, d in restricciones:
    modelo += f"{a:g}x1 + {b:g}x2 {signo} {d:g}\n"
modelo += "x1, x2 >= 0"
st.code(modelo, language=None)

if not resolver:
    st.info("Ingrese los datos y presione Resolver problema.")
    st.stop()

# ---- Simplex ----
simplex = MetodoSimplex(tipo, c1, c2, restricciones)
resultado = simplex.resolver()
estado = resultado["estado"]

st.subheader("Sistema transformado")
st.write("<= suma holgura (s); >= resta superavit (e) y suma artificial (A); "
         "= suma artificial (A). Las artificiales se penalizan con M = 1 000 000.")
for ecuacion in resultado["ecuaciones"]:
    st.write("•", ecuacion)

st.subheader("Iteraciones del metodo Simplex")
columnas = resultado["nombres"] + ["LD", "Razon"]

for it in resultado["iteraciones"]:
    titulo = "Tabla inicial" if it["numero"] == 0 else f"Iteracion {it['numero']}"
    if it["entra"] != "-":
        titulo += f" | Entra: {it['entra']}"
    if it["sale"] != "-":
        titulo += f" | Sale: {it['sale']}"

    with st.expander(titulo, expanded=(it["numero"] == 0)):
        razones = []
        for r in it["razones"]:
            razones.append(None if r == float("inf") else r)
        while len(razones) < 4:
            razones.append(None)          # la fila Cj - Zj no tiene razon

        tabla = pd.DataFrame(np.round(it["tabla"], 4) + 0.0,
                             index=it["base"] + ["Cj - Zj"],
                             columns=resultado["nombres"] + ["LD"])
        tabla["Razon"] = razones
        st.dataframe(tabla)

        if it["pivote"] is not None:
            st.write(f"**Entra:** {it['entra']} | **Sale:** {it['sale']} | "
                     f"**Pivote:** {it['pivote']:.4f}")
            st.write("Operaciones de Gauss-Jordan:")
            for operacion in it["operaciones"]:
                st.write("•", operacion)

if tipo == "Minimizar":
    st.caption("En minimizacion se trabaja con -Z; por eso el ultimo valor de "
               "la fila Cj - Zj es -Z.")

# ---- Resultado ----
st.subheader("Resultado del metodo Simplex")
if estado == "Optimo":
    st.success("Solucion optima encontrada")
    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("x1", f"{resultado['x1']:.6g}")
    col_2.metric("x2", f"{resultado['x2']:.6g}")
    col_3.metric("Z optimo", f"{resultado['z']:.6g}")

    extras = []
    for nombre, valor in resultado["valores"].items():
        if nombre not in ["x1", "x2"] and not nombre.startswith("A"):
            extras.append((nombre, valor))
    st.write("Variables de holgura y superavit:")
    st.dataframe(pd.DataFrame(extras, columns=["Variable", "Valor"]))
elif estado == "Infactible":
    st.error("El problema es INFACTIBLE")
elif estado == "No acotado":
    st.warning("El problema es NO ACOTADO")
else:
    st.error(estado)

st.subheader("Interpretacion matematica")
for linea in simplex.interpretar(resultado):
    st.write("•", linea)

# ---- Metodo grafico ----
grafico = MetodoGrafico(tipo, c1, c2, restricciones)
datos, optimo = grafico.evaluar_vertices()
st.subheader("Metodo grafico")
st.pyplot(grafico.dibujar(datos, optimo if estado == "Optimo" else None))

if len(datos) > 0:
    st.write("Vertices factibles y Z en cada uno:")
    st.dataframe(pd.DataFrame(datos, columns=["x1", "x2", "Z"]))
else:
    st.write("No hay vertices factibles: la region factible es vacia.")

# ---- Comparacion ----
st.subheader("Comparacion Simplex vs grafico")
if estado == "Optimo" and optimo is not None:
    st.dataframe(pd.DataFrame(
        [["Simplex", resultado["x1"], resultado["x2"], resultado["z"]],
         ["Grafico", optimo[0], optimo[1], optimo[2]]],
        columns=["Metodo", "x1", "x2", "Z"]))
    mismo_z = abs(resultado["z"] - optimo[2]) < TOL
    mismo_punto = (abs(resultado["x1"] - optimo[0]) < TOL
                   and abs(resultado["x2"] - optimo[1]) < TOL)
    if mismo_z and mismo_punto:
        st.success("Ambos metodos coinciden: mismo punto y mismo valor de Z.")
    elif mismo_z:
        st.info("Mismo Z en vertices distintos: hay soluciones optimas alternativas.")
    else:
        st.error("Los metodos NO coinciden. Revise los datos.")
elif estado == "Infactible":
    st.info("Ambos metodos concuerdan: no hay region factible, no hay solucion.")
elif estado == "No acotado":
    st.info("La region se extiende sin limite en la direccion que mejora Z, "
            "por eso no se marca ningun optimo.")
