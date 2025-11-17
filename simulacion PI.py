# -*- coding: utf-8 -*-
"""
Simulación en 'tiempo real' del control PI de temperatura en Data Center.
Planta térmica + actuador + sensor + PI con feedforward.

Se muestran 4 gráficos:
1) Temperatura real y medida vs setpoint
2) Potencia de frío (u_cmd y u_eff)
3) Carga térmica y temperatura ambiente
4) Error de control y umbrales

Hay dos botones:
- "Pico carga": suma +heat_amp kW durante heat_dur pasos.
- "Ola calor": suma +amb_amp °C durante amb_dur pasos.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ======================================
# Parámetros de tiempo
# ======================================
Ts = 10.0
T_total = 6000.0
N = int(T_total / Ts)

# ======================================
# Setpoint fijo
# ======================================
T_ref = 24.0

# ======================================
# Planta / actuador / sensor
# ======================================
T_amb_base = 28.0

tau_th  = 300.0
Kh      = 0.0002
Kc_th   = -0.00035

tau_act = 60.0
U_max   = 100.0
U_min   = 0.0

tau_sens = 60.0

# ======================================
# Controlador PI
# ======================================
Kp = -2.0
Ti = 600.0
Ki = Kp / Ti
Ki_d = Ki * Ts

use_feedforward = True

# Umbrales para graficar
e_low  = 0.5
e_mid  = 1.5
e_high = 3.0

# ======================================
# Carga térmica y ruido
# ======================================
q_nom = 25.0
# Si querés que al inicio esté PERFECTAMENTE plano, podés poner 0.0
q_spread = 0.10  

# Perturbaciones interactivas
heat_amp = 500.0
heat_dur = 4
amb_amp  = 10.0
amb_dur  = 4

heat_pulse_steps = 0
amb_pulse_steps  = 0

# ======================================
# Callbacks botones
# ======================================
def on_heat(event):
    global heat_pulse_steps
    heat_pulse_steps = heat_dur
    print(f">> Pico de carga térmica (+{heat_amp} kW)")

def on_amb(event):
    global amb_pulse_steps
    amb_pulse_steps = amb_dur
    print(f">> Ola de calor (+{amb_amp} °C)")

# ======================================
# Estados — Arranque en equilibrio térmico
# ======================================
T      = [T_ref]      # Temperatura REAL arrancando EXACTO en 24°C
T_meas = [T_ref]      # Sensor perfectamente alineado desde el inicio
e      = [0.0]        # Sin error inicial

# --- Cálculo CORRECTO de u_eq usando la dinámica de la planta ---
# 0 = -(T_ref - T_amb_base)/tau_th + Kh*q_nom + Kc_th*u_eq
num  = (T_ref - T_amb_base)/tau_th - Kh * q_nom
u_eq = num / Kc_th
u_eq = np.clip(u_eq, U_min, U_max)

# --- Integral inicial para que el PI + feedforward también estén en equilibrio ---
# u_eq = q_nom + Kp*0 + I_eq => I_eq = u_eq - q_nom
I_eq = u_eq - (q_nom if use_feedforward else 0.0)

u_cmd  = [u_eq]        # Comando estable inicial
u_eff  = [u_eq]        # Actuador arrancando ya estabilizado
Iterm  = [I_eq]        # Integral inicial ajustada al equilibrio

q_hist    = [q_nom]
Tamb_hist = [T_amb_base]
t_hist    = [0.0]

# ======================================
# Figura y subplots
# ======================================
plt.ion()
fig = plt.figure(figsize=(13, 10))

ax1 = plt.subplot2grid((5,4), (0,0), colspan=4)   # temperatura
ax2 = plt.subplot2grid((5,4), (1,0), colspan=4)   # acción de control
ax3 = plt.subplot2grid((5,4), (2,0), colspan=4)   # perturbaciones
ax4 = plt.subplot2grid((5,4), (3,0), colspan=4)   # error
ax_btn_heat = plt.subplot2grid((5,4), (4,0), colspan=2)
ax_btn_amb  = plt.subplot2grid((5,4), (4,2), colspan=2)

# Temperatura
line_T,    = ax1.plot([], [], label='T real')
line_Tm,   = ax1.plot([], [], '--', label='T medida')
line_Tref, = ax1.plot([], [], ':', label='Setpoint 24°C')
ax1.set_ylabel('°C'); ax1.grid(True); ax1.legend()

# Control
line_ucmd, = ax2.plot([], [], label='u_cmd [kW]')
line_ueff, = ax2.plot([], [], '--', label='u_eff [kW]')
ax2.set_ylabel('kW'); ax2.grid(True); ax2.legend()

# Perturbaciones
line_q,    = ax3.plot([], [], label='Carga térmica [kW]')
line_Tamb, = ax3.plot([], [], '--', label='T_amb [°C]')
ax3.set_ylabel('q/T'); ax3.grid(True); ax3.legend()

# Error
line_err, = ax4.plot([], [], label='Error e(t)')
ax4.axhline(e_low, linestyle=':')
ax4.axhline(-e_low, linestyle=':')
ax4.axhline(e_mid, linestyle=':')
ax4.axhline(-e_mid, linestyle=':')
ax4.axhline(e_high, linestyle=':')
ax4.axhline(-e_high, linestyle=':')
ax4.set_ylabel('°C'); ax4.set_xlabel('Tiempo [s]')
ax4.grid(True); ax4.legend()

# Botones
btn_heat = Button(ax_btn_heat, "Pico carga")
btn_amb  = Button(ax_btn_amb,  "Ola calor")
btn_heat.on_clicked(on_heat)
btn_amb.on_clicked(on_amb)

fig.tight_layout()
plt.show()

# ======================================
# Bucle de simulación con actualización
# ======================================
for k in range(1, N):

    t_k = k * Ts

    # Carga térmica base + ruido
    qk = q_nom * (1 + q_spread * (2*np.random.rand() - 1))

    if heat_pulse_steps > 0:
        qk += heat_amp
        heat_pulse_steps -= 1

    # Ambiente
    Tamb_k = T_amb_base
    if amb_pulse_steps > 0:
        Tamb_k += amb_amp
        amb_pulse_steps -= 1

    # Sensor
    if tau_sens > 0:
        Tm_k = T_meas[-1] + (Ts/tau_sens)*(T[-1] - T_meas[-1])
    else:
        Tm_k = T[-1]

    # Error
    e_k = T_ref - Tm_k

    # Feedforward
    u_ff = qk if use_feedforward else 0.0

    # PI sin saturar
    u_unsat = u_ff + Kp*e_k + Iterm[-1]

    # Saturación
    u_sat = np.clip(u_unsat, U_min, U_max)

    # Anti-windup
    if U_min < u_unsat < U_max:
        I_k = Iterm[-1] + Ki_d*e_k
    else:
        I_k = Iterm[-1]

    # Actuador
    if tau_act > 0:
        ueff_k = u_eff[-1] + (Ts/tau_act)*(u_sat - u_eff[-1])
    else:
        ueff_k = u_sat

    # Dinámica térmica
    dT = (-(T[-1] - Tamb_k)/tau_th + Kh*qk + Kc_th*ueff_k)*Ts
    T_k = T[-1] + dT

    # Guardamos
    t_hist.append(t_k)
    T.append(T_k)
    T_meas.append(Tm_k)
    u_cmd.append(u_sat)
    u_eff.append(ueff_k)
    e.append(e_k)
    Iterm.append(I_k)
    q_hist.append(qk)
    Tamb_hist.append(Tamb_k)

    # Actualizar gráficos
    line_T.set_data(t_hist, T)
    line_Tm.set_data(t_hist, T_meas)
    line_Tref.set_data(t_hist, [T_ref]*len(t_hist))

    line_ucmd.set_data(t_hist, u_cmd)
    line_ueff.set_data(t_hist, u_eff)

    line_q.set_data(t_hist, q_hist)
    line_Tamb.set_data(t_hist, Tamb_hist)

    line_err.set_data(t_hist, e)

    # Límites
    ax1.set_xlim(0, max(600, t_k))
    ax1.set_ylim(20, 30)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(0, U_max + 20)

    ax3.set_xlim(ax1.get_xlim())
    ax3.set_ylim(0, max(max(q_hist)+10, max(Tamb_hist)+5))

    ax4.set_xlim(ax1.get_xlim())
    ax4.set_ylim(-5, 5)

    plt.pause(0.03)

plt.ioff()
plt.show()
