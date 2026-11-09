import numpy as np
from grafico import MetodoGrafico

M = 1000000      # constante M del metodo de la Gran M
TOL = 0.000001   # tolerancia para comparar con cero


class MetodoSimplex:
    """Simplex con Gran M. Restriccion = tupla (a, b, signo, derecho)."""

    def __init__(self, tipo, c1, c2, restricciones):
        self.tipo = tipo
        self.c1 = c1
        self.c2 = c2
        self.restricciones = restricciones

    def construir_tabla(self):
        """Convierte las restricciones en ecuaciones y arma la tabla inicial."""

        # Si el lado derecho es negativo se multiplica la restriccion por -1
        normalizadas = []
        for a, b, signo, d in self.restricciones:
            if d < 0:
                a, b, d = -a, -b, -d
                if signo == "<=":
                    signo = ">="
                elif signo == ">=":
                    signo = "<="
            normalizadas.append((a, b, signo, d))

        nombres = ["x1", "x2"]
        base = []
        ecuaciones = []
        extras_por_fila = []

        for i, (a, b, signo, d) in enumerate(normalizadas, start=1):
            if signo == "<=":                      # holgura
                extras = [(f"s{i}", 1)]
                base.append(f"s{i}")
                texto = f"{a:g}x1 + {b:g}x2 + s{i} = {d:g}"
            elif signo == ">=":                    # superavit + artificial
                extras = [(f"e{i}", -1), (f"A{i}", 1)]
                base.append(f"A{i}")
                texto = f"{a:g}x1 + {b:g}x2 - e{i} + A{i} = {d:g}"
            else:                                  # igualdad: artificial
                extras = [(f"A{i}", 1)]
                base.append(f"A{i}")
                texto = f"{a:g}x1 + {b:g}x2 + A{i} = {d:g}"

            for nombre, coeficiente in extras:
                nombres.append(nombre)
            extras_por_fila.append(extras)
            ecuaciones.append(texto)

        # Tabla: una fila por restriccion, ultima columna = lado derecho
        tabla = np.zeros((3, len(nombres) + 1))
        for fila, (a, b, signo, d) in enumerate(normalizadas):
            tabla[fila, 0] = a
            tabla[fila, 1] = b
            tabla[fila, -1] = d
            for nombre, coeficiente in extras_por_fila[fila]:
                tabla[fila, nombres.index(nombre)] = coeficiente

        # Se trabaja siempre maximizando: minimizar Z = maximizar -Z
        signo_z = 1 if self.tipo == "Maximizar" else -1
        objetivo = np.zeros(len(nombres))
        objetivo[0] = signo_z * self.c1
        objetivo[1] = signo_z * self.c2
        for j, nombre in enumerate(nombres):
            if nombre.startswith("A"):
                objetivo[j] = -M                   # penalizacion de la artificial

        return tabla, nombres, base, objetivo, ecuaciones, normalizadas

    def resolver(self):
        tabla, nombres, base, objetivo, ecuaciones, normalizadas = self.construir_tabla()
        iteraciones = []

        for numero in range(51):
            # Fila Cj - Zj:  Zj = (coeficientes de la base) * (tabla)
            cb = np.array([objetivo[nombres.index(v)] for v in base])
            cj_zj = objetivo - cb @ tabla[:, :-1]
            valor = cb @ tabla[:, -1]
            tabla_completa = np.vstack([tabla.copy(), np.append(cj_zj, valor)])

            valores = {nombre: 0.0 for nombre in nombres}
            for fila, v in enumerate(base):
                valores[v] = tabla[fila, -1]
            artificial_positiva = False
            for nombre in nombres:
                if nombre.startswith("A") and valores[nombre] > TOL:
                    artificial_positiva = True

            iteracion = {"numero": numero, "tabla": tabla_completa,
                         "base": base.copy(), "entra": "-", "sale": "-",
                         "razones": [], "pivote": None, "operaciones": []}
            iteraciones.append(iteracion)
            resultado = {"iteraciones": iteraciones, "ecuaciones": ecuaciones,
                         "nombres": nombres, "valores": valores,
                         "normalizadas": normalizadas, "x1": 0, "x2": 0, "z": 0}

            # 1) Si ningun Cj - Zj es positivo, termino
            if np.max(cj_zj) <= TOL:
                if artificial_positiva:
                    resultado["estado"] = "Infactible"
                else:
                    resultado["estado"] = "Optimo"
                    resultado["x1"] = valores["x1"]
                    resultado["x2"] = valores["x2"]
                    resultado["z"] = self.c1 * valores["x1"] + self.c2 * valores["x2"]
                return resultado

            # 2) Variable que entra: el Cj - Zj mas grande
            col = int(np.argmax(cj_zj))
            iteracion["entra"] = nombres[col]

            # 3) Razones = lado derecho / coeficiente (solo si coeficiente > 0)
            razones = []
            for fila in range(3):
                if tabla[fila, col] > TOL:
                    razones.append(tabla[fila, -1] / tabla[fila, col])
                else:
                    razones.append(float("inf"))
            iteracion["razones"] = razones

            # Si no hay razones, Z crece sin limite (o no hay region factible)
            if min(razones) == float("inf"):
                if artificial_positiva and len(MetodoGrafico(self.tipo, self.c1, self.c2, self.restricciones).calcular_vertices()) == 0:
                    resultado["estado"] = "Infactible"
                else:
                    resultado["estado"] = "No acotado"
                return resultado

            # 4) Variable que sale: la menor razon
            fila_pivote = int(np.argmin(razones))
            pivote = tabla[fila_pivote, col]
            iteracion["sale"] = base[fila_pivote]
            iteracion["pivote"] = pivote
            iteracion["fila_pivote"] = fila_pivote
            iteracion["col_pivote"] = col

            # 5) Gauss-Jordan: pivote = 1 y el resto de la columna = 0
            iteracion["operaciones"].append(
                f"F{fila_pivote + 1} ← F{fila_pivote + 1} / {pivote:.4f}"
            )
            tabla[fila_pivote] = tabla[fila_pivote] / pivote
            for fila in range(3):
                if fila != fila_pivote:
                    factor = tabla[fila, col]
                    if abs(factor) > TOL:
                        iteracion["operaciones"].append(
                            f"F{fila + 1} ← F{fila + 1} − ({factor:.4f}) · F{fila_pivote + 1}"
                        )
                    tabla[fila] = tabla[fila] - factor * tabla[fila_pivote]
            base[fila_pivote] = nombres[col]

        resultado["estado"] = "Limite de iteraciones"
        return resultado


    def interpretar(self, resultado):
        estado = resultado["estado"]

        if estado == "Infactible":
            return ["Ningun punto cumple todas las restricciones a la vez: la "
                    "region factible es vacia y el problema no tiene solucion."]

        if estado == "No acotado":
            mejora = "crecer" if self.tipo == "Maximizar" else "disminuir"
            return [f"Se puede mover la solucion sin salir de la region factible "
                    f"y Z puede {mejora} sin limite: no existe un optimo finito."]

        if estado != "Optimo":
            return ["No se llego a un resultado en el maximo de iteraciones."]

        lineas = [f"El optimo es x1 = {resultado['x1']:.6g}, "
                  f"x2 = {resultado['x2']:.6g}, Z = {resultado['z']:.6g}."]
        for i, (a, b, signo, d) in enumerate(resultado["normalizadas"], start=1):
            if signo == "=":
                lineas.append(f"R{i} es una igualdad: se cumple con igualdad.")
                continue
            nombre = f"s{i}" if signo == "<=" else f"e{i}"
            valor = resultado["valores"][nombre]
            if valor < TOL:
                lineas.append(f"R{i}: {nombre} = 0, la restriccion esta activa "
                              "(se cumple con igualdad en el optimo).")
            else:
                lineas.append(f"R{i}: {nombre} = {valor:.6g}, la restriccion no "
                              "esta activa (sobra ese margen).")
        return lineas
