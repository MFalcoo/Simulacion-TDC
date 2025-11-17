# -*- coding: utf-8 -*-
"""
Simulación: Control de Temperatura en Data Center (lazo cerrado)
Planta térmica + actuador HVAC (1er orden) + sensor (1er orden)
Control PI (P + I discreto con anti-windup) + feedforward opcional.

Qué graficamos:
1) Temperatura real vs. medida vs. setpoint
2) Potencia de frío (comando y efectiva)
3) Carga térmica y temperatura ambiente
4) Error y umbrales

Autor: (tu nombre)
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1) PARÁMETROS BÁSICOS DE TIEMPO
# ============================================================

Ts = 30.0          # [s] tiempo de muestreo
T_total = 6_000.0  # [s] duración total (~100 min)
N = int(T_total / Ts)

t = np.arange(N) * Ts   # eje de tiempo [s]

# ============================================================
# 2) SETPOINT FIJO 
# ============================================================

T_ref = 24.0              # [°C] setpoint fijo
T_ref_profile = np.full(N, T_ref)

# ============================================================
# 3) PLANTA, ACTUADOR, SENSOR
# ============================================================

T_amb_base = 28.0         # [°C] temperatura ambiente base

tau_th  = 300.0           # [s] constante de tiempo térmica (~5 min)
Kh      = 0.0002          # [°C/(kW·s)]
Kc_th   = -0.00035        # [°C/(kW·s)]

tau_act = 60.0            # [s]
U_max   = 100.0           # [kW]
U_min   = 0.0             # [kW]

tau_sens = 60.0           # [s] retardo del sensor

# ============================================================
# 4) CONTROLADOR PI DISCRETO
# ============================================================

Kp = -2.0         # Proporcional (e>0 => se reduce el frío)
Ti = 600.0        # [s] tiempo integral
Ki = Kp / Ti      # continuo
Ki_d = Ki * Ts    # discreto

e_low  = 0.5
e_mid  = 1.5
e_high = 3.0

use_feedforward = True

# ============================================================
# 5) CARGA TÉRMICA Y PERTURBACIONES — Interactivo 
# ============================================================

q_nom = 25.0
q_spread = 0.10

# Valores por defecto
heat_amp = 20.0
heat_dur = 3
heat_t_min = 20.0

amb_amp = 2.0
amb_dur = 3
amb_t_min = 50.0

# --- Interactivo ---
try:
    ans2 = input("¿Querés configurar perturbaciones manualmente? (s/n) [n]: ").strip().lower()
except EOFError:
    ans2 = "n"

if ans2 == "s":
    try:
        heat_amp = float(input("Pico de carga térmica [kW] (ej 20): ") or heat_amp)
        heat_dur = int(input("Duración pico carga [muestras] (ej 3): ") or heat_dur)
        heat_t_min = float(input("Inicio pico carga [min] (ej 20): ") or heat_t_min)

        amb_amp = float(input("Pico de temperatura ambiente [°C] (ej 2): ") or amb_amp)
        amb_dur = int(input("Duración pico ambiente [muestras] (ej 3): ") or amb_dur)
        amb_t_min = float(input("Inicio pico ambiente [min] (ej 50): ") or amb_t_min)
    except ValueError:
        print("Entrada inválida, se usan parámetros por defecto.")

k_heat = int((heat_t_min * 60.0) / Ts)
k_amb  = int((amb_t_min * 60.0) / Ts)

def build_profile_load():
    q = np.full(N, q_nom, dtype=float)
    # Ruido térmico de servers
    rnd = 1.0 + q_spread * (2.0*np.random.rand(N) - 1.0)
    q *= rnd
    # pico
    if 0 <= k_heat < N:
        q[k_heat:k_heat+heat_dur] += heat_amp
    return np.clip(q, 0.0, None)

def build_profile_ambient():
    Tamb = np.full(N, T_amb_base, dtype=float)
    if 0 <= k_amb < N:
        Tamb[k_amb:k_amb+amb_dur] += amb_amp
    return Tamb

q_profile = build_profile_load()
Tamb_profile = build_profile_ambient()

# ============================================================
# 6) VARIABLES DE ESTADO
# ============================================================

T      = np.zeros(N)
T_meas = np.zeros(N)
u_cmd  = np.zeros(N)
u_eff  = np.zeros(N)
e      = np.zeros(N)
Iterm  = np.zeros(N)

T[0] = max(T_ref, T_amb_base + 1.0)
T_meas[0] = T[0]
u_eff[0] = 0.0

# ============================================================
# 7) SIMULACIÓN
# ============================================================

for k in range(1, N):

    # Sensor
    if tau_sens > 0:
        T_meas[k] = T_meas[k-1] + (Ts / tau_sens) * (T[k-1] - T_meas[k-1])
    else:
        T_meas[k] = T[k-1]

    # Error
    e[k] = T_ref - T_meas[k]

    # Feedforward
    qk = q_profile[k]
    Tamb_k = Tamb_profile[k]
    u_ff = qk if use_feedforward else 0.0

    # PI sin saturar
    u_unsat = u_ff + Kp * e[k] + Iterm[k-1]

    # Saturación
    u_sat = np.clip(u_unsat, U_min, U_max)

    # Anti-windup
    if U_min < u_unsat < U_max:
        Iterm[k] = Iterm[k-1] + Ki_d * e[k]
    else:
        Iterm[k] = Iterm[k-1]

    u_cmd[k] = u_sat

    # Actuador
    if tau_act > 0:
        u_eff[k] = u_eff[k-1] + (Ts / tau_act) * (u_cmd[k] - u_eff[k-1])
    else:
        u_eff[k] = u_cmd[k]

    # Dinámica térmica
    dT = (-(T[k-1] - Tamb_k) / tau_th + Kh * qk + Kc_th * u_eff[k]) * Ts
    T[k] = T[k-1] + dT

# ============================================================
# 8) REPORT
# ============================================================

print("----- Resumen -----")
print(f"Setpoint fijo: {T_ref:.1f} °C")
print(f"Kp = {Kp}, Ti = {Ti} s")
print(f"Perturbación térmica: +{heat_amp} kW en t ≈ {heat_t_min} min")
print(f"Perturbación ambiente: +{amb_amp} °C en t ≈ {amb_t_min} min")

# ============================================================
# 9) GRÁFICOS
# ============================================================

plt.figure(figsize=(12, 9))

# 1) Temperatura
plt.subplot(4,1,1)
plt.plot(t, T, label='T real')
plt.plot(t, T_meas, '--', label='T medida')
plt.plot(t, T_ref_profile, ':', label='Setpoint 24°C')
plt.ylabel('°C')
plt.title('Temperatura de sala')
plt.grid(True); plt.legend()

# 2) Control
plt.subplot(4,1,2)
plt.plot(t, u_cmd, label='u_cmd')
plt.plot(t, u_eff, '--', label='u_eff')
plt.ylabel('kW')
plt.title('Acción de control')
plt.grid(True); plt.legend()

# 3) Perturbaciones
plt.subplot(4,1,3)
plt.plot(t, q_profile, label='Carga térmica [kW]')
plt.plot(t, Tamb_profile, '--', label='T_amb [°C]')
plt.ylabel('q / Tamb')
plt.title('Perturbaciones')
plt.grid(True); plt.legend()

# 4) Error
plt.subplot(4,1,4)
plt.plot(t, e, label='Error')
plt.axhline(+e_low, linestyle=':')
plt.axhline(-e_low, linestyle=':')
plt.axhline(+e_mid, linestyle=':')
plt.axhline(-e_mid, linestyle=':')
plt.axhline(+e_high, linestyle=':')
plt.axhline(-e_high, linestyle=':')
plt.xlabel('Tiempo [s]')
plt.ylabel('°C')
plt.title('Error de control')
plt.grid(True); plt.legend()

plt.tight_layout()
plt.savefig("sim_dc_PI_final.png", dpi=170)
plt.show()
