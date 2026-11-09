import itertools
import numpy as np
import matplotlib.pyplot as plt

TOL = 0.000001   # tolerancia para comparar con cero


class MetodoGrafico:
    """Metodo grafico: vertices, region factible y optimo."""

    def __init__(self, tipo, c1, c2, restricciones):
        self.tipo = tipo
        self.c1 = c1
        self.c2 = c2
        self.restricciones = restricciones

    def punto_factible(self, x1, x2):
        if x1 < -TOL or x2 < -TOL:
            return False
        for a, b, signo, d in self.restricciones:
            valor = a * x1 + b * x2
            if signo == "<=" and valor > d + TOL:
                return False
            if signo == ">=" and valor < d - TOL:
                return False
            if signo == "=" and abs(valor - d) > TOL:
                return False
        return True

    def calcular_vertices(self):
        """Interseca todas las rectas de dos en dos y se queda con las factibles."""
        rectas = list(self.restricciones)
        rectas.append((1, 0, "=", 0))   # eje x2 (x1 = 0)
        rectas.append((0, 1, "=", 0))   # eje x1 (x2 = 0)
        vertices = []

        for r1, r2 in itertools.combinations(rectas, 2):
            matriz = np.array([[r1[0], r1[1]], [r2[0], r2[1]]], dtype=float)
            if abs(np.linalg.det(matriz)) < 1e-10:
                continue                # rectas paralelas
            x1, x2 = np.linalg.solve(matriz, [r1[3], r2[3]])
            if self.punto_factible(x1, x2):
                punto = (max(0.0, float(x1)), max(0.0, float(x2)))
                repetido = False
                for v in vertices:
                    if abs(v[0] - punto[0]) < TOL and abs(v[1] - punto[1]) < TOL:
                        repetido = True
                if not repetido:
                    vertices.append(punto)
        return vertices

    def evaluar_vertices(self):
        """Evalua Z en cada vertice y devuelve tambien el mejor."""
        datos = []
        for x1, x2 in self.calcular_vertices():
            datos.append((x1, x2, self.c1 * x1 + self.c2 * x2))
        if len(datos) == 0:
            return datos, None
        if self.tipo == "Maximizar":
            mejor = max(datos, key=lambda p: p[2])
        else:
            mejor = min(datos, key=lambda p: p[2])
        return datos, mejor

    def dibujar(self, datos, optimo):
        limite = 10
        for a, b, signo, d in self.restricciones:
            limite = max(limite, abs(d) / max(abs(a), abs(b), 1e-9))
        for x1, x2, z in datos:
            limite = max(limite, x1, x2)
        limite = limite * 1.35

        fig, ejes = plt.subplots(figsize=(7, 5))
        x = np.linspace(0, limite, 500)

        # Rectas de las restricciones
        for i, (a, b, signo, d) in enumerate(self.restricciones, start=1):
            etiqueta = f"R{i}: {a:g}x1 + {b:g}x2 {signo} {d:g}"
            if abs(b) > TOL:
                ejes.plot(x, (d - a * x) / b, label=etiqueta)
            elif abs(a) > TOL:
                ejes.axvline(d / a, label=etiqueta)

        # Region factible: se pintan los puntos de una malla que cumplen todo
        hay_igualdad = False
        malla_x, malla_y = np.meshgrid(x, x)
        factible = np.ones(malla_x.shape, dtype=bool)
        for a, b, signo, d in self.restricciones:
            valor = a * malla_x + b * malla_y
            if signo == "<=":
                factible = factible & (valor <= d + TOL)
            elif signo == ">=":
                factible = factible & (valor >= d - TOL)
            else:
                hay_igualdad = True

        if hay_igualdad:
            # con una igualdad la region es un segmento entre los vertices
            if len(datos) >= 2:
                puntos = sorted(datos)
                ejes.plot([p[0] for p in puntos], [p[1] for p in puntos],
                          color="tab:blue", linewidth=5, alpha=0.4)
        elif factible.any():
            ejes.imshow(np.where(factible, 1.0, np.nan), origin="lower",
                        extent=(0, limite, 0, limite), cmap="Blues",
                        vmin=0, vmax=2, alpha=0.45, aspect="auto")

        # Vertices
        for x1, x2, z in datos:
            ejes.scatter(x1, x2, color="black", zorder=5)
            ejes.annotate(f"({x1:.2f}, {x2:.2f})", (x1, x2),
                          xytext=(5, 5), textcoords="offset points")

        # Optimo y recta de nivel Z = valor optimo
        if optimo is not None:
            ejes.scatter(optimo[0], optimo[1], s=160, color="red",
                         marker="*", zorder=6, label="Optimo grafico")
            if abs(self.c2) > TOL:
                ejes.plot(x, (optimo[2] - self.c1 * x) / self.c2, "--", color="red",
                          linewidth=1.2, label=f"Z = {optimo[2]:.4g}")

        ejes.set_xlim(0, limite)
        ejes.set_ylim(0, limite)
        ejes.set_xlabel("x1")
        ejes.set_ylabel("x2")
        ejes.set_title("Metodo grafico")
        ejes.grid(alpha=0.25)
        ejes.legend(fontsize=8)
        fig.tight_layout()
        return fig
