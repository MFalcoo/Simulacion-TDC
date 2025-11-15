# -*- coding: utf-8 -*-
"""
Simulación: Control de Temperatura en Data Center (lazo cerrado)
Planta térmica + actuador HVAC (1er orden) + sensor (1er orden)
+ Control PI (+ feedforward opcional)

Qué graficamos:
1) Temperatura real de sala vs. medida vs. setpoint
2) Potencia de frío (comando y efectiva)
3) Carga térmica (kW) y temperatura ambiente
4) Error de control y umbrales

Autor: (tu nombre)
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Parámetros "de TP"
# ==========================

# Tiempo de muestreo y duración
Ts = 30.0                   # [s] muestreo
T_total = 6_000.0           # [s] ~100 min
N = int(T_total / Ts)

# Setpoint y entorno
T_ref = 24.0                # [°C] consigna
T_amb_base = 28.0           # [°C] temperatura ambiente base (sala externa)

# Planta térmica (primer orden)
tau_th = 300.0              # [s] constante de tiempo térmica de sala (≈5 min)
Kh = 0.0002                 # [°C/(kW·s)] efecto de carga térmica integrada
Kc_th = -0.00035            # [°C/(kW·s)] efecto de HVAC (negativo: más frío → baja T)

# Actuador HVAC (primer orden)
tau_act = 60.0              # [s] inercia del actuador/compresor/ventiladores
U_max = 100.0               # [kW] potencia de frío máxima
U_min = 0.0                 # [kW]
use_feedforward = True      # feedforward basado en q(t)

# Sensor (primer orden)
tau_sens = 60.0             # [s] retardo/filtrado del sensor

# Controlador PI (e = T_ref - T_meas)
# e > 0 → sala "fría" respecto a consigna → hay que bajar el frío ⇒ Kp, Ki negativos
Kp = -2.0                   # [kW/°C]
Ki = -0.004                 # [kW/(°C·s)]  <-- ajustable
I_min, I_max = -50.0, 50.0  # límites al término integral (anti-windup simple)

# Umbrales de “estado operativo” (solo para graficar)
e_low  = 0.5                # [°C] ±0.5 (zona normalidad)
e_mid  = 1.5
e_high = 3.0

# ==========================
# Escenarios / señales externas
# ==========================

# Carga térmica nominal (promedio) y variaciones
q_nom = 25.0                # [kW] carga base (servidores)
q_spread = 0.10             # +/-10% aleatorio

# Perturbaciones de carga (picos tipo “DoS térmico”): (instante, +kW, duración en pasos)
heat_spikes = [
    (40,  +20.0, 3),        # ~ 20 min: +20 kW por 3 muestras
    (55,  +25.0, 3),        # ~ 27.5 min
    (70,  +30.0, 3),        # ~ 35 min
]

# Perturbación de ambiente (ola de calor / falla ventilación): (instante, +°C, duración en pasos)
ambient_spikes = [
    (20, +2.0, 3),          # ~ 10 min
    (85, +3.0, 4),          # ~ 42.5 min
]

# ==========================
# Helpers de perturbaciones
# ==========================

def build_profile(N, base, spikes, Ts, is_heat=True):
    """Devuelve una señal de longitud N con base y eventos tipo 'spike'."""
    sig = np.full(N, base, dtype=float)
    for (t_idx, inc, dur) in spikes:
        t_idx = int(t_idx)  # índice en muestras
        t_end = min(N, t_idx + int(dur))
        sig[t_idx:t_end] += inc
    if is_heat:
        # ruido/microvariación para q(t)
        rnd = 1.0 + q_spread * (2.0*np.random.rand(N) - 1.0)
        sig *= rnd
        sig = np.clip(sig, 0.0, None)
    return sig

# Perfiles discretos:
q_profile = build_profile(N, q_nom, heat_spikes, Ts, is_heat=True)             # [kW]
Tamb_profile = build_profile(N, T_amb_base, ambient_spikes, Ts, is_heat=False) # [°C]

# ==========================
# Simulación
# ==========================

# Estados
T = np.zeros(N)             # [°C] temperatura de sala (real)
T_meas = np.zeros(N)        # [°C] temperatura medida (sensor)
u_cmd = np.zeros(N)         # [kW] comando del controlador PI (+ feedforward)
u_eff = np.zeros(N)         # [kW] potencia de frío efectiva (salida del actuador)
e = np.zeros(N)             # [°C] error
I = np.zeros(N)             # término integral del PI
zones = []                  # zona de error (NORMAL / BAJO / MEDIO / ALTO)

# Inicialización
T[0] = max(T_ref, T_amb_base + 1.0)   # arrancamos un poco por arriba del SP
T_meas[0] = T[0]
u_eff[0] = 0.0

for k in range(1, N):
    # Señales externas
    qk = q_profile[k]            # [kW]
    Tamb_k = Tamb_profile[k]     # [°C]

    # Dinámica del sensor (1er orden hacia T)
    if tau_sens > 0:
        T_meas[k] = T_meas[k-1] + (Ts/tau_sens)*(T[k-1] - T_meas[k-1])
    else:
        T_meas[k] = T[k-1]

    # Error
    e[k] = T_ref - T_meas[k]

    # Zona/umbrales (logging)
    ae = abs(e[k])
    if ae <= e_low:
        zone = "NORMAL"
    elif ae <= e_mid:
        zone = "BAJO"
    elif ae <= e_high:
        zone = "MEDIO"
    else:
        zone = "ALTO"
    zones.append(zone)

    # Feedforward: compensar aproximadamente la carga térmica qk
    u_ff = qk if use_feedforward else 0.0

    # --- Control PI ---
    # Parte proporcional
    P = Kp * e[k]

    # Parte integral (sumador de Euler)
    I[k] = I[k-1] + Ki * e[k] * Ts

    # Anti-windup: limitar el término integral
    I[k] = np.clip(I[k], I_min, I_max)

    # Comando "sin límites" (feedforward + PI)
    u_cmd_raw = u_ff + P + I[k]

    # Refuerzo de compresor si el error es muy grande
    comp_on = (ae > e_high)
    if comp_on:
        u_cmd_raw = max(u_cmd_raw, 0.8 * U_max)

    # Saturación física del actuador
    u_cmd[k] = np.clip(u_cmd_raw, U_min, U_max)

    # Dinámica del actuador (1er orden hacia u_eff)
    if tau_act > 0:
        u_eff[k] = u_eff[k-1] + (Ts/tau_act) * (u_cmd[k] - u_eff[k-1])
    else:
        u_eff[k] = u_cmd[k]

    # Dinámica térmica de la sala (Euler hacia adelante)
    # dT/dt = -(T - Tamb)/tau_th + Kh*qk + Kc_th*u_eff
    dT = (-(T[k-1] - Tamb_k)/tau_th + Kh*qk + Kc_th*u_eff[k]) * Ts
    T[k] = T[k-1] + dT

# ==========================
# Reporte por consola
# ==========================
print("----- Resumen de simulación -----")
print(f"Muestras: {N}  |  Ts = {Ts:.1f} s  |  Duración = {N*Ts/60:.1f} min")
print(f"Setpoint: {T_ref:.1f} °C  |  Umax = {U_max:.1f} kW")
print(f"Ganancias PI: Kp = {Kp:.3f}, Ki = {Ki:.5f}")
print(f"Planta: tau_th = {tau_th:.1f} s, Kh = {Kh:.5f}, Kc_th = {Kc_th:.5f}")
print(f"Actuador: tau_act = {tau_act:.1f} s  |  Sensor: tau_sens = {tau_sens:.1f} s")

# ==========================
# Gráficos para el informe
# ==========================

t = np.arange(N) * Ts

plt.figure(figsize=(12, 9))

plt.subplot(4,1,1)
plt.plot(t, T, label='T sala (real)')
plt.plot(t, T_meas, linestyle='--', label='T medida (sensor)')
plt.axhline(T_ref, linestyle=':', label='Setpoint')
plt.ylabel('Temperatura [°C]')
plt.title('Temperatura de sala vs. medida')
plt.grid(True); plt.legend()

plt.subplot(4,1,2)
plt.plot(t, u_cmd, label='u_cmd (kW)')
plt.plot(t, u_eff, linestyle='--', label='u_eff (kW)')
plt.ylabel('Potencia de frío [kW]')
plt.title('Acción de control (comando vs. efectiva)')
plt.grid(True); plt.legend()

plt.subplot(4,1,3)
plt.plot(t, q_profile, label='Carga térmica q(t) [kW]')
plt.plot(t, Tamb_profile, linestyle='--', label='T_amb(t) [°C]')
plt.ylabel('q [kW] / T_amb [°C]')
plt.title('Perturbaciones: carga térmica y ambiente')
plt.grid(True); plt.legend()

plt.subplot(4,1,4)
plt.plot(t, e, label='Error e(t) = Tref - Tmeas')
plt.axhline(+e_low, linestyle=':', label='±0.5 °C'); plt.axhline(-e_low, linestyle=':')
plt.axhline(+e_mid, linestyle=':'); plt.axhline(-e_mid, linestyle=':')
plt.axhline(+e_high, linestyle=':'); plt.axhline(-e_high, linestyle=':')
plt.xlabel('Tiempo [s]')
plt.ylabel('Error [°C]')
plt.title('Error de control y umbrales')
plt.grid(True); plt.legend()

plt.tight_layout()
plt.savefig('sim_dc_control_PI.png', dpi=160)
plt.show()
