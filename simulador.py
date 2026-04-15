import tkinter as tk
from tkinter import ttk
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# =========================================================
# 1. SISTEMA 1: RIESGO
# =========================================================

# Entradas
meteo = ctrl.Antecedent(
    np.arange(0, 10.1, 0.1), "meteo"
)  # 0 = muy mala, 10 = muy buena
road = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "road")  # 0 = muy mala, 10 = muy buena
distance = ctrl.Antecedent(
    np.arange(0, 100.1, 0.1), "distance"
)  # metros al coche delantero

# Salida
risk = ctrl.Consequent(np.arange(0, 10.1, 0.1), "risk")

# Funciones de pertenencia
meteo["bad"] = fuzz.trimf(meteo.universe, [0, 0, 5])
meteo["medium"] = fuzz.trimf(meteo.universe, [2.5, 5, 7.5])
meteo["good"] = fuzz.trimf(meteo.universe, [5, 10, 10])

road["bad"] = fuzz.trimf(road.universe, [0, 0, 5])
road["medium"] = fuzz.trimf(road.universe, [2.5, 5, 7.5])
road["good"] = fuzz.trimf(road.universe, [5, 10, 10])

distance["close"] = fuzz.trimf(distance.universe, [0, 0, 35])
distance["medium"] = fuzz.trimf(distance.universe, [20, 50, 80])
distance["far"] = fuzz.trimf(distance.universe, [60, 100, 100])

risk["low"] = fuzz.trimf(risk.universe, [0, 0, 4.5])
risk["medium"] = fuzz.trimf(risk.universe, [3, 5, 7])
risk["high"] = fuzz.trimf(risk.universe, [5.5, 10, 10])

# Reglas del riesgo
# Casos claramente peligrosos
r1 = ctrl.Rule(distance["close"], risk["high"])
r2 = ctrl.Rule(meteo["bad"] & road["bad"], risk["high"])
r3 = ctrl.Rule(meteo["bad"] & distance["medium"], risk["high"])
r4 = ctrl.Rule(road["bad"] & distance["medium"], risk["high"])
r5 = ctrl.Rule(meteo["bad"] & distance["close"], risk["high"])
r6 = ctrl.Rule(road["bad"] & distance["close"], risk["high"])
r7 = ctrl.Rule(meteo["medium"] & road["bad"] & distance["medium"], risk["high"])
r8 = ctrl.Rule(meteo["bad"] & road["medium"] & distance["medium"], risk["high"])

# Casos intermedios
r9 = ctrl.Rule(meteo["bad"], risk["medium"])
r10 = ctrl.Rule(road["bad"], risk["medium"])
r11 = ctrl.Rule(distance["medium"], risk["medium"])
r12 = ctrl.Rule(meteo["medium"] & road["medium"], risk["medium"])
r13 = ctrl.Rule(meteo["medium"] & distance["medium"], risk["medium"])
r14 = ctrl.Rule(road["medium"] & distance["medium"], risk["medium"])

# Casos favorables
r15 = ctrl.Rule(meteo["good"] & road["good"] & distance["far"], risk["low"])
r16 = ctrl.Rule(meteo["good"] & distance["far"], risk["low"])
r17 = ctrl.Rule(road["good"] & distance["far"], risk["low"])
# r18 = ctrl.Rule(meteo["good"] & road["good"], risk["low"])

risk_ctrl = ctrl.ControlSystem(
    [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17]
)

# =========================================================
# 2. SISTEMA 2: VELOCIDAD RECOMENDADA
# =========================================================

# Entradas
risk_in = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "risk_in")
speed_limit = ctrl.Antecedent(np.arange(20, 120.1, 0.1), "speed_limit")
driving_style = ctrl.Antecedent(
    np.arange(0, 10.1, 0.1), "driving_style"
)  # 0 prudente, 10 agresivo

# Salida
recommended_speed = ctrl.Consequent(np.arange(0, 120.1, 0.1), "recommended_speed")

# Funciones de pertenencia
risk_in["low"] = fuzz.trimf(risk_in.universe, [0, 0, 4.5])
risk_in["medium"] = fuzz.trimf(risk_in.universe, [3, 5, 7])
risk_in["high"] = fuzz.trimf(risk_in.universe, [5.5, 10, 10])

speed_limit["very_low"] = fuzz.trimf(speed_limit.universe, [20, 20, 40])
speed_limit["low"] = fuzz.trimf(speed_limit.universe, [30, 45, 60])
speed_limit["medium"] = fuzz.trimf(speed_limit.universe, [50, 70, 90])
speed_limit["high"] = fuzz.trimf(speed_limit.universe, [80, 95, 110])
speed_limit["very_high"] = fuzz.trimf(speed_limit.universe, [100, 120, 120])

driving_style["prudent"] = fuzz.trimf(driving_style.universe, [0, 0, 4])
driving_style["normal"] = fuzz.trimf(driving_style.universe, [2.5, 5, 7.5])
driving_style["aggressive"] = fuzz.trimf(driving_style.universe, [6, 10, 10])

recommended_speed["very_low"] = fuzz.trimf(recommended_speed.universe, [0, 0, 30])
recommended_speed["low"] = fuzz.trimf(recommended_speed.universe, [20, 40, 60])
recommended_speed["medium"] = fuzz.trimf(recommended_speed.universe, [50, 70, 90])
recommended_speed["high"] = fuzz.trimf(recommended_speed.universe, [80, 100, 120])
recommended_speed["very_high"] = fuzz.trimf(recommended_speed.universe, [110, 120, 120])

v1 = ctrl.Rule(
    speed_limit["very_low"] & risk_in["high"] & driving_style["prudent"],
    recommended_speed["very_low"],
)
v2 = ctrl.Rule(
    speed_limit["very_low"] & risk_in["high"] & driving_style["normal"],
    recommended_speed["very_low"],
)
v3 = ctrl.Rule(
    speed_limit["very_low"] & risk_in["high"] & driving_style["aggressive"],
    recommended_speed["very_low"],
)

v4 = ctrl.Rule(
    speed_limit["low"] & risk_in["high"] & driving_style["prudent"],
    recommended_speed["very_low"],
)
v5 = ctrl.Rule(
    speed_limit["low"] & risk_in["high"] & driving_style["normal"],
    recommended_speed["very_low"],
)
v6 = ctrl.Rule(
    speed_limit["low"] & risk_in["high"] & driving_style["aggressive"],
    recommended_speed["low"],
)

v7 = ctrl.Rule(
    speed_limit["medium"] & risk_in["high"] & driving_style["prudent"],
    recommended_speed["very_low"],
)
v8 = ctrl.Rule(
    speed_limit["medium"] & risk_in["high"] & driving_style["normal"],
    recommended_speed["low"],
)
v9 = ctrl.Rule(
    speed_limit["medium"] & risk_in["high"] & driving_style["aggressive"],
    recommended_speed["low"],
)

v10 = ctrl.Rule(
    speed_limit["high"] & risk_in["high"] & driving_style["prudent"],
    recommended_speed["low"],
)
v11 = ctrl.Rule(
    speed_limit["high"] & risk_in["high"] & driving_style["normal"],
    recommended_speed["low"],
)
v12 = ctrl.Rule(
    speed_limit["high"] & risk_in["high"] & driving_style["aggressive"],
    recommended_speed["medium"],
)

v13 = ctrl.Rule(
    speed_limit["very_high"] & risk_in["high"] & driving_style["prudent"],
    recommended_speed["low"],
)
v14 = ctrl.Rule(
    speed_limit["very_high"] & risk_in["high"] & driving_style["normal"],
    recommended_speed["low"],
)
v15 = ctrl.Rule(
    speed_limit["very_high"] & risk_in["high"] & driving_style["aggressive"],
    recommended_speed["medium"],
)

v16 = ctrl.Rule(
    speed_limit["very_low"] & risk_in["medium"] & driving_style["prudent"],
    recommended_speed["very_low"],
)
v17 = ctrl.Rule(
    speed_limit["very_low"] & risk_in["medium"] & driving_style["normal"],
    recommended_speed["very_low"],
)
v18 = ctrl.Rule(
    speed_limit["very_low"] & risk_in["medium"] & driving_style["aggressive"],
    recommended_speed["low"],
)

v19 = ctrl.Rule(
    speed_limit["low"] & risk_in["medium"] & driving_style["prudent"],
    recommended_speed["low"],
)
v20 = ctrl.Rule(
    speed_limit["low"] & risk_in["medium"] & driving_style["normal"],
    recommended_speed["low"],
)
v21 = ctrl.Rule(
    speed_limit["low"] & risk_in["medium"] & driving_style["aggressive"],
    recommended_speed["medium"],
)

v22 = ctrl.Rule(
    speed_limit["medium"] & risk_in["medium"] & driving_style["prudent"],
    recommended_speed["low"],
)
v23 = ctrl.Rule(
    speed_limit["medium"] & risk_in["medium"] & driving_style["normal"],
    recommended_speed["medium"],
)
v24 = ctrl.Rule(
    speed_limit["medium"] & risk_in["medium"] & driving_style["aggressive"],
    recommended_speed["high"],
)

v25 = ctrl.Rule(
    speed_limit["high"] & risk_in["medium"] & driving_style["prudent"],
    recommended_speed["medium"],
)
v26 = ctrl.Rule(
    speed_limit["high"] & risk_in["medium"] & driving_style["normal"],
    recommended_speed["high"],
)
v27 = ctrl.Rule(
    speed_limit["high"] & risk_in["medium"] & driving_style["aggressive"],
    recommended_speed["high"],
)

v28 = ctrl.Rule(
    speed_limit["very_high"] & risk_in["medium"] & driving_style["prudent"],
    recommended_speed["medium"],
)
v29 = ctrl.Rule(
    speed_limit["very_high"] & risk_in["medium"] & driving_style["normal"],
    recommended_speed["high"],
)
v30 = ctrl.Rule(
    speed_limit["very_high"] & risk_in["medium"] & driving_style["aggressive"],
    recommended_speed["high"],
)

v31 = ctrl.Rule(
    speed_limit["very_low"] & risk_in["low"] & driving_style["prudent"],
    recommended_speed["very_low"],
)
v32 = ctrl.Rule(
    speed_limit["very_low"] & risk_in["low"] & driving_style["normal"],
    recommended_speed["low"],
)
v33 = ctrl.Rule(
    speed_limit["very_low"] & risk_in["low"] & driving_style["aggressive"],
    recommended_speed["low"],
)

v34 = ctrl.Rule(
    speed_limit["low"] & risk_in["low"] & driving_style["prudent"],
    recommended_speed["low"],
)
v35 = ctrl.Rule(
    speed_limit["low"] & risk_in["low"] & driving_style["normal"],
    recommended_speed["medium"],
)
v36 = ctrl.Rule(
    speed_limit["low"] & risk_in["low"] & driving_style["aggressive"],
    recommended_speed["high"],
)

v37 = ctrl.Rule(
    speed_limit["medium"] & risk_in["low"] & driving_style["prudent"],
    recommended_speed["medium"],
)
v38 = ctrl.Rule(
    speed_limit["medium"] & risk_in["low"] & driving_style["normal"],
    recommended_speed["medium"],
)
v39 = ctrl.Rule(
    speed_limit["medium"] & risk_in["low"] & driving_style["aggressive"],
    recommended_speed["high"],
)

v40 = ctrl.Rule(
    speed_limit["high"] & risk_in["low"] & driving_style["prudent"],
    recommended_speed["medium"],
)
v41 = ctrl.Rule(
    speed_limit["high"] & risk_in["low"] & driving_style["normal"],
    recommended_speed["high"],
)
v42 = ctrl.Rule(
    speed_limit["high"] & risk_in["low"] & driving_style["aggressive"],
    recommended_speed["very_high"],
)

v43 = ctrl.Rule(
    speed_limit["very_high"] & risk_in["low"] & driving_style["prudent"],
    recommended_speed["high"],
)
v44 = ctrl.Rule(
    speed_limit["very_high"] & risk_in["low"] & driving_style["normal"],
    recommended_speed["very_high"],
)
v45 = ctrl.Rule(
    speed_limit["very_high"] & risk_in["low"] & driving_style["aggressive"],
    recommended_speed["very_high"],
)

speed_ctrl = ctrl.ControlSystem(
    [
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        v8,
        v9,
        v10,
        v11,
        v12,
        v13,
        v14,
        v15,
        v16,
        v17,
        v18,
        v19,
        v20,
        v21,
        v22,
        v23,
        v24,
        v25,
        v26,
        v27,
        v28,
        v29,
        v30,
        v31,
        v32,
        v33,
        v34,
        v35,
        v36,
        v37,
        v38,
        v39,
        v40,
        v41,
        v42,
        v43,
        v44,
        v45,
    ]
)

# =========================================================
# 3. FUNCIONES AUXILIARES
# =========================================================


def compute_risk(meteo_value, road_value, distance_value):
    sim = ctrl.ControlSystemSimulation(risk_ctrl)
    sim.input["meteo"] = meteo_value
    sim.input["road"] = road_value
    sim.input["distance"] = distance_value
    sim.compute()
    return sim.output["risk"]


def compute_speed(risk_value, speed_limit_value, driving_style_value):
    sim = ctrl.ControlSystemSimulation(speed_ctrl)
    sim.input["risk_in"] = risk_value
    sim.input["speed_limit"] = speed_limit_value
    sim.input["driving_style"] = driving_style_value
    sim.compute()
    return sim.output["recommended_speed"]


def get_main_fuzzy_label(fuzzy_var, value):
    memberships = {}
    for label in fuzzy_var.terms:
        memberships[label] = fuzz.interp_membership(
            fuzzy_var.universe, fuzzy_var[label].mf, value
        )
    best_label = max(memberships, key=memberships.get)
    return str(best_label).upper()


def get_color_for_risk_class(risk_class):
    if risk_class == "LOW":
        return "green"
    elif risk_class == "MEDIUM":
        return "orange"
    return "red"


# =========================================================
# INTERFAZ
# =========================================================


class FuzzyCarSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Fuzzy Car Speed Simulator")
        self.root.geometry("1200x700")
        self.root.configure(bg="white")

        self.real_speed = 0.0
        self.recommended_speed = 0.0
        self.risk_value = 0.0
        self.car_x = 80

        self.dt_ms = 50
        self.alpha = 0.08

        self.build_ui()
        self.update_simulation()

    def build_ui(self):
        control_frame = tk.Frame(self.root, bg="white", padx=15, pady=15)
        control_frame.pack(side="left", fill="y")

        title = tk.Label(
            control_frame, text="Inputs", font=("Arial", 18, "bold"), bg="white"
        )
        title.pack(anchor="w", pady=(0, 15))

        self.meteo_var = tk.DoubleVar(value=8.0)
        self.road_var = tk.DoubleVar(value=8.0)
        self.distance_var = tk.DoubleVar(value=60.0)
        self.speed_limit_var = tk.DoubleVar(value=90.0)
        self.driving_style_var = tk.DoubleVar(value=5.0)

        self.create_slider(control_frame, "Clima", self.meteo_var, 0, 10, 0.1)
        self.meteo_class_label = tk.Label(
            control_frame, text="Clase: good", font=("Arial", 11), bg="white"
        )
        self.meteo_class_label.pack(anchor="w", pady=(0, 8))

        self.create_slider(
            control_frame, "Estado de la carretera", self.road_var, 0, 10, 0.1
        )
        self.road_class_label = tk.Label(
            control_frame, text="Clase: good", font=("Arial", 11), bg="white"
        )
        self.road_class_label.pack(anchor="w", pady=(0, 8))

        self.create_slider(
            control_frame,
            "Distancia al coche delantero (m)",
            self.distance_var,
            0,
            100,
            1,
        )
        self.distance_class_label = tk.Label(
            control_frame, text="Clase: far", font=("Arial", 11), bg="white"
        )
        self.distance_class_label.pack(anchor="w", pady=(0, 8))

        self.create_slider(
            control_frame,
            "Límite de velocidad (km/h)",
            self.speed_limit_var,
            20,
            120,
            1,
        )
        self.speed_limit_class_label = tk.Label(
            control_frame, text="Clase: medium", font=("Arial", 11), bg="white"
        )
        self.speed_limit_class_label.pack(anchor="w", pady=(0, 8))

        self.create_slider(
            control_frame, "Estilo de conducción", self.driving_style_var, 0, 10, 0.1
        )
        self.driving_style_class_label = tk.Label(
            control_frame, text="Clase: normal", font=("Arial", 11), bg="white"
        )
        self.driving_style_class_label.pack(anchor="w", pady=(0, 8))

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=15)

        self.risk_label = tk.Label(
            control_frame, text="Riesgo: 0.00", font=("Arial", 14), bg="white"
        )
        self.risk_label.pack(anchor="w", pady=5)

        self.risk_class_label = tk.Label(
            control_frame, text="Clase: medium", font=("Arial", 11, "bold"), bg="white"
        )
        self.risk_class_label.pack(anchor="w", pady=(0, 8))

        self.rec_speed_label = tk.Label(
            control_frame,
            text="Velocidad recomendada: 0.00 km/h",
            font=("Arial", 14),
            bg="white",
        )
        self.rec_speed_label.pack(anchor="w", pady=5)

        self.real_speed_label = tk.Label(
            control_frame,
            text="Velocidad real: 0.00 km/h",
            font=("Arial", 14),
            bg="white",
        )
        self.real_speed_label.pack(anchor="w", pady=5)

        display_frame = tk.Frame(self.root, bg="white")
        display_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(
            display_frame, width=760, height=500, bg="#eaf4ff", highlightthickness=0
        )
        self.canvas.pack(pady=10)

        self.speed_bar_canvas = tk.Canvas(
            display_frame, width=760, height=80, bg="white", highlightthickness=0
        )
        self.speed_bar_canvas.pack()

    def create_slider(self, parent, label_text, variable, min_val, max_val, resolution):
        frame = tk.Frame(parent, bg="white")
        frame.pack(fill="x", pady=8)

        label = tk.Label(frame, text=label_text, font=("Arial", 12), bg="white")
        label.pack(anchor="w")

        scale = tk.Scale(
            frame,
            variable=variable,
            from_=min_val,
            to=max_val,
            orient="horizontal",
            resolution=resolution,
            length=300,
            bg="white",
            highlightthickness=0,
        )
        scale.pack(anchor="w")

    def draw_scene(self):
        self.canvas.delete("all")

        # carretera
        self.canvas.create_rectangle(0, 220, 760, 360, fill="#444", outline="")
        self.canvas.create_line(0, 290, 760, 290, fill="white", width=3, dash=(20, 20))

        # cielo / césped
        self.canvas.create_rectangle(0, 0, 760, 220, fill="#cfe8ff", outline="")
        self.canvas.create_rectangle(0, 360, 760, 500, fill="#a8d08d", outline="")

        # coche azul (ego)
        self.canvas.create_rectangle(
            self.car_x,
            250,
            self.car_x + 90,
            320,
            fill="dodgerblue",
            outline="black",
            width=2,
        )
        self.canvas.create_text(
            self.car_x + 45,
            235,
            text="Ego car",
            font=("Arial", 10, "bold"),
            fill="white",
        )

        # coche delantero rojo
        distance_value = self.distance_var.get()
        visual_gap = 100 + (distance_value / 100) * 250
        front_x = self.car_x + visual_gap
        if front_x > 620:
            front_x = 620

        self.canvas.create_rectangle(
            front_x, 250, front_x + 90, 320, fill="red", outline="black", width=2
        )
        self.canvas.create_text(
            front_x + 45,
            235,
            text="Front car",
            font=("Arial", 10, "bold"),
            fill="white",
        )

        self.canvas.create_text(
            380,
            390,
            text=f"Distance to front car: {distance_value:.1f} m",
            font=("Arial", 14, "bold"),
        )

    def draw_speed_bar(self):
        self.speed_bar_canvas.delete("all")

        self.speed_bar_canvas.create_text(80, 20, text="0", font=("Arial", 10))
        self.speed_bar_canvas.create_text(680, 20, text="120", font=("Arial", 10))
        self.speed_bar_canvas.create_rectangle(
            60, 30, 700, 55, outline="black", width=2
        )

        # velocidad real
        real_width = (self.real_speed / 120) * 640
        self.speed_bar_canvas.create_rectangle(
            60, 30, 60 + real_width, 55, fill="dodgerblue", outline=""
        )

        # velocidad recomendada
        rec_x = 60 + (self.recommended_speed / 120) * 640
        self.speed_bar_canvas.create_line(rec_x, 25, rec_x, 60, fill="green", width=3)

        # límite
        lim_x = 60 + (self.speed_limit_var.get() / 120) * 640
        self.speed_bar_canvas.create_line(
            lim_x, 25, lim_x, 60, fill="red", width=3, dash=(4, 2)
        )

        self.speed_bar_canvas.create_text(
            180,
            70,
            text=f"Velocidad real: {self.real_speed:.1f} km/h",
            font=("Arial", 18),
        )
        self.speed_bar_canvas.create_text(
            430,
            70,
            text=f"Recomendada: {self.recommended_speed:.1f} km/h",
            font=("Arial", 18),
            fill="green",
        )
        self.speed_bar_canvas.create_text(
            630,
            70,
            text=f"Límite: {self.speed_limit_var.get():.1f} km/h",
            font=("Arial", 18),
            fill="red",
        )

    def update_simulation(self):
        meteo_value = self.meteo_var.get()
        road_value = self.road_var.get()
        distance_value = self.distance_var.get()
        speed_limit_value = self.speed_limit_var.get()
        driving_style_value = self.driving_style_var.get()

        # Fuzzy principal
        self.risk_value = compute_risk(meteo_value, road_value, distance_value)
        self.recommended_speed = compute_speed(
            self.risk_value, speed_limit_value, driving_style_value
        )

        # por seguridad, también limitamos aquí
        # self.recommended_speed = min(self.recommended_speed, speed_limit_value)

        # Dinámica del coche
        self.real_speed = self.real_speed + self.alpha * (
            self.recommended_speed - self.real_speed
        )

        # Clases principales
        meteo_class = get_main_fuzzy_label(meteo, meteo_value)
        road_class = get_main_fuzzy_label(road, road_value)
        distance_class = get_main_fuzzy_label(distance, distance_value)
        speed_limit_class = get_main_fuzzy_label(speed_limit, speed_limit_value)
        driving_style_class = get_main_fuzzy_label(driving_style, driving_style_value)
        risk_class = get_main_fuzzy_label(risk, self.risk_value)

        risk_color = get_color_for_risk_class(risk_class)

        # Actualizar textos
        self.meteo_class_label.config(text=f"Clase: {meteo_class}")
        self.road_class_label.config(text=f"Clase: {road_class}")
        self.distance_class_label.config(text=f"Clase: {distance_class}")
        self.speed_limit_class_label.config(text=f"Clase: {speed_limit_class}")
        self.driving_style_class_label.config(text=f"Clase: {driving_style_class}")

        self.risk_label.config(text=f"Riesgo: {self.risk_value:.2f}", fg=risk_color)
        self.risk_class_label.config(text=f"Clase: {risk_class}", fg=risk_color)

        self.rec_speed_label.config(
            text=f"Velocidad recomendada: {self.recommended_speed:.2f} km/h"
        )
        self.real_speed_label.config(text=f"Velocidad real: {self.real_speed:.2f} km/h")

        # Movimiento visual
        self.car_x += self.real_speed * 0.02
        if self.car_x > 140:
            self.car_x = 80

        self.draw_scene()
        self.draw_speed_bar()

        self.root.after(self.dt_ms, self.update_simulation)


import matplotlib.pyplot as plt


def save_membership_plot(fuzzy_var, title, xlabel, filename):
    plt.figure(figsize=(8, 4.5))

    for label in fuzzy_var.terms:
        plt.plot(fuzzy_var.universe, fuzzy_var[label].mf, linewidth=2, label=str(label))

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Grado de pertenencia", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def export_membership_images():
    save_membership_plot(
        meteo,
        "Función de pertenencia - Meteorología",
        "Meteorología",
        "meteo_membership.png",
    )

    save_membership_plot(
        road,
        "Función de pertenencia - Estado de la carretera",
        "Estado de la carretera",
        "road_membership.png",
    )

    save_membership_plot(
        distance,
        "Función de pertenencia - Distancia frontal",
        "Distancia frontal (m)",
        "distance_membership.png",
    )
    save_membership_plot(
        risk_in, "Funciones de pertenencia - Riesgo", "Riesgo", "riesgo_membership.png"
    )

    save_membership_plot(
        driving_style,
        "Funciones de pertenencia - Estilo de conducción",
        "Estilo de conducción",
        "estilo_conduccion_membership.png",
    )

    save_membership_plot(
        speed_limit,
        "Funciones de pertenencia - Límite de velocidad",
        "Velocidad (km/h)",
        "limite_velocidad_membership.png",
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    # export_membership_images()
    root = tk.Tk()
    app = FuzzyCarSimulator(root)
    root.mainloop()
