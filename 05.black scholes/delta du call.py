import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. La fonction prend maintenant un TABLEAU 'S' complet en entrée
def calculate_call_delta_vectorized(S, K, T, r, sigma):
    if T <= 0:
        return np.where(S > K, 1.0, 0.0)
    # Plus besoin de boucler, NumPy gère le tableau 'S' d'un coup
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# --- PARAMÈTRES ---
K = 100
r = 0.05
S_range = np.linspace(50, 150, 500)

# --- CONFIGURATION STYLE NÉON ---
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
bg_color = "#0B0E14"
fig.patch.set_facecolor(bg_color)
ax1.set_facecolor(bg_color)
ax2.set_facecolor(bg_color)
neon_colors = ["#00F0FF", "#00FF66", "#FF007F"]

# --- GRAPHIQUE 1 : VECTORISÉ ---
vol_fixe = 0.20
maturites = [1.0, 0.3, 0.05]

for T, color in zip(maturites, neon_colors):
    # ICI : On passe directement S_range (sans boucle for !)
    deltas = calculate_call_delta_vectorized(S_range, K, T, r, vol_fixe)
    ax1.plot(S_range, deltas, label=f'T = {T} an(s)', color=color, linewidth=2)

ax1.axvline(x=K, color='#FF5E00', linestyle='--', alpha=0.8, label='Strike (K=100)')
ax1.set_title("Effet du Temps sur le Delta (Vol = 20%)", color="#FFFFFF", fontsize=12, pad=15)
ax1.set_xlabel("Prix de l'action (S)", color="#8E9AA6")
ax1.set_ylabel("Delta du Call", color="#8E9AA6")
ax1.grid(True, color="#1F242E", linestyle='-', linewidth=0.5)
ax1.legend(facecolor=bg_color, edgecolor="#1F242E")

# --- GRAPHIQUE 2 : VECTORISÉ ---
T_fixe = 0.5
volatilites = [0.10, 0.30, 0.60]

for sigma, color in zip(volatilites, neon_colors):
    # ICI AUSSI : Envoi direct du tableau
    deltas = calculate_call_delta_vectorized(S_range, K, T_fixe, r, sigma)
    ax2.plot(S_range, deltas, label=f'Vol = {sigma*100:.0f}%', color=color, linewidth=2)

ax2.axvline(x=K, color='#FF5E00', linestyle='--', alpha=0.8, label='Strike (K=100)')
ax2.set_title("Effet de la Volatilité sur le Delta (T = 6 mois)", color="#FFFFFF", fontsize=12, pad=15)
ax2.set_xlabel("Prix de l'action (S)", color="#8E9AA6")
ax2.set_ylabel("Delta du Call", color="#8E9AA6")
ax2.grid(True, color="#1F242E", linestyle='-', linewidth=0.5)
ax2.legend(facecolor=bg_color, edgecolor="#1F242E")

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_color("#1F242E")
    ax.tick_params(colors="#8E9AA6")

plt.tight_layout()
plt.show()