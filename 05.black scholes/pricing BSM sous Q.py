import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# PARAMETRES DU MARCHE
# Calibres sur nos donnees SP500
# ============================================================

S = 7365.46   # Prix actuel SP500 (dernier prix connu)
r = 0.04      # Taux sans risque (OAT 10 ans)
sigma = 0.1626  # Volatilite empirique annualisee

# ============================================================
# FORMULE BLACK-SCHOLES
# ============================================================

def black_scholes(S, K, T, r, sigma, type_option='call'):
    """
    Prix d'une option europeenne par Black-Scholes

    S     : prix actuel du sous-jacent
    K     : strike
    T     : maturite en annees
    r     : taux sans risque
    sigma : volatilite annualisee
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type_option == 'call':
        prix = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        prix = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return prix, d1, d2

# ============================================================
# GREEKS
# ============================================================

def greeks(S, K, T, r, sigma, type_option='call'):
    """
    Calcul des Greeks d'une option europeenne
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta : dC/dS — de combien bouge l option si S bouge de 1
    if type_option == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma : d²C/dS² — de combien bouge le delta si S bouge de 1
    # Identique pour call et put
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega : dC/dsigma — de combien bouge l option si sigma bouge de 1%
    # Identique pour call et put
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # divise par 100 pour exprimer en % de vol

    # Theta : dC/dT — perte de valeur par jour
    if type_option == 'call':
        theta = (- S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252
    else:
        theta = (- S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252

    # Rho : dC/dr — sensibilite au taux sans risque
    if type_option == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return delta, gamma, vega, theta, rho

# ============================================================
# EXEMPLE CONCRET
# On price un call et un put ATM (at the money) a 3 mois
# ============================================================

K = 7365.46   # ATM — strike = prix actuel
T = 3/12      # 3 mois

prix_call, d1, d2 = black_scholes(S, K, T, r, sigma, 'call')
prix_put,  _,  _  = black_scholes(S, K, T, r, sigma, 'put')

delta_c, gamma_c, vega_c, theta_c, rho_c = greeks(S, K, T, r, sigma, 'call')
delta_p, gamma_p, vega_p, theta_p, rho_p = greeks(S, K, T, r, sigma, 'put')

print("=" * 60)
print("MODULE 5 — BLACK-SCHOLES PRICING")
print("=" * 60)
print(f"Sous-jacent (S)          : {S:.2f}")
print(f"Strike (K)               : {K:.2f}  (ATM)")
print(f"Maturite (T)             : {T*12:.0f} mois")
print(f"Taux sans risque (r)     : {r:.2%}")
print(f"Volatilite (sigma)       : {sigma:.2%}")
print(f"d1                       : {d1:.4f}")
print(f"d2                       : {d2:.4f}")
print()
print(f"{'':25} {'CALL':>10} {'PUT':>10}")
print("-" * 50)
print(f"{'Prix BS':25} {prix_call:>10.2f} {prix_put:>10.2f}")
print(f"{'Delta':25} {delta_c:>10.4f} {delta_p:>10.4f}")
print(f"{'Gamma':25} {gamma_c:>10.6f} {gamma_p:>10.6f}")
print(f"{'Vega (par 1% de vol)':25} {vega_c:>10.4f} {vega_p:>10.4f}")
print(f"{'Theta (par jour)':25} {theta_c:>10.4f} {theta_p:>10.4f}")
print(f"{'Rho (par 1% de taux)':25} {rho_c:>10.4f} {rho_p:>10.4f}")
print()

# Verification parite call-put
# C - P = S - K * exp(-rT)
parite = prix_call - prix_put
theorique = S - K * np.exp(-r * T)
print(f"Parite call-put C-P      : {parite:.4f}")
print(f"S - K*exp(-rT)           : {theorique:.4f}")
print(f"Verification             : {'OK' if abs(parite - theorique) < 0.01 else 'ERREUR'}")

# ============================================================
# SURFACE DE PRIX — DIFFERENTS STRIKES ET MATURITES
# ============================================================

strikes    = np.linspace(S * 0.80, S * 1.20, 50)  # de -20% a +20%
maturites  = [1/12, 3/12, 6/12, 12/12]             # 1, 3, 6, 12 mois
labels_mat = ['1 mois', '3 mois', '6 mois', '12 mois']

plt.style.use('dark_background')
bg_color   = "#0B0E14"
cyan       = "#00F0FF"
green      = "#00FF66"
pink       = "#FF007F"
orange     = "#FF8C00"
yellow     = "#FFD700"
grid_color = "#1E2A38"
couleurs   = [cyan, green, orange, pink]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor(bg_color)

# --- Prix call ---
ax = axes[0, 0]
ax.set_facecolor(bg_color)
for mat, label, col in zip(maturites, labels_mat, couleurs):
    prix_calls = [black_scholes(S, k, mat, r, sigma, 'call')[0] for k in strikes]
    ax.plot(strikes, prix_calls, color=col, linewidth=2, label=label)
ax.axvline(x=S, color='white', linewidth=0.8, linestyle='--', alpha=0.5, label='ATM')
ax.set_title('Prix Call', color=cyan, fontsize=12, fontweight='bold')
ax.set_xlabel('Strike', color='white')
ax.set_ylabel('Prix', color='white')
ax.tick_params(colors='white')
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)

# --- Prix put ---
ax = axes[0, 1]
ax.set_facecolor(bg_color)
for mat, label, col in zip(maturites, labels_mat, couleurs):
    prix_puts = [black_scholes(S, k, mat, r, sigma, 'put')[0] for k in strikes]
    ax.plot(strikes, prix_puts, color=col, linewidth=2, label=label)
ax.axvline(x=S, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_title('Prix Put', color=cyan, fontsize=12, fontweight='bold')
ax.set_xlabel('Strike', color='white')
ax.set_ylabel('Prix', color='white')
ax.tick_params(colors='white')
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)

# --- Delta ---
ax = axes[0, 2]
ax.set_facecolor(bg_color)
for mat, label, col in zip(maturites, labels_mat, couleurs):
    deltas = [greeks(S, k, mat, r, sigma, 'call')[0] for k in strikes]
    ax.plot(strikes, deltas, color=col, linewidth=2, label=label)
ax.axvline(x=S, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axhline(y=0.5, color='white', linewidth=0.5, linestyle=':', alpha=0.4)
ax.set_title('Delta Call', color=cyan, fontsize=12, fontweight='bold')
ax.set_xlabel('Strike', color='white')
ax.set_ylabel('Delta', color='white')
ax.tick_params(colors='white')
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)

# --- Gamma ---
ax = axes[1, 0]
ax.set_facecolor(bg_color)
for mat, label, col in zip(maturites, labels_mat, couleurs):
    gammas = [greeks(S, k, mat, r, sigma, 'call')[1] for k in strikes]
    ax.plot(strikes, gammas, color=col, linewidth=2, label=label)
ax.axvline(x=S, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_title('Gamma', color=cyan, fontsize=12, fontweight='bold')
ax.set_xlabel('Strike', color='white')
ax.set_ylabel('Gamma', color='white')
ax.tick_params(colors='white')
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)

# --- Vega ---
ax = axes[1, 1]
ax.set_facecolor(bg_color)
for mat, label, col in zip(maturites, labels_mat, couleurs):
    vegas = [greeks(S, k, mat, r, sigma, 'call')[2] for k in strikes]
    ax.plot(strikes, vegas, color=col, linewidth=2, label=label)
ax.axvline(x=S, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_title('Vega (par 1% vol)', color=cyan, fontsize=12, fontweight='bold')
ax.set_xlabel('Strike', color='white')
ax.set_ylabel('Vega', color='white')
ax.tick_params(colors='white')
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)

# --- Theta ---
ax = axes[1, 2]
ax.set_facecolor(bg_color)
for mat, label, col in zip(maturites, labels_mat, couleurs):
    thetas = [greeks(S, k, mat, r, sigma, 'call')[3] for k in strikes]
    ax.plot(strikes, thetas, color=col, linewidth=2, label=label)
ax.axvline(x=S, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_title('Theta (par jour)', color=cyan, fontsize=12, fontweight='bold')
ax.set_xlabel('Strike', color='white')
ax.set_ylabel('Theta', color='white')
ax.tick_params(colors='white')
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)

fig.suptitle(
    f'Black-Scholes   |   S={S:.0f}   r={r:.2%}   sigma={sigma:.2%}   Call ATM={prix_call:.2f}   Put ATM={prix_put:.2f}',
    color=yellow, fontsize=12, fontweight='bold'
)

plt.tight_layout()
plt.savefig('sp500_black_scholes.png', dpi=150, bbox_inches='tight', facecolor=bg_color)
plt.show()
print("Graphique sauvegarde : sp500_black_scholes.png")