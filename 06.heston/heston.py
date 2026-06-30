import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

# ============================================================
# PARAMETRES HESTON
# Calibres typiquement sur marche SP500
# ============================================================

S     = 7365.46   # Prix actuel
K     = 7365.46   # Strike ATM
T     = 3/12      # 3 mois
r     = 0.04      # Taux sans risque
q     = 0.0       # Dividende

# Parametres Heston
v0    = 0.1626**2  # Variance initiale (sigma empirique au carre)
kappa = 2.0        # Vitesse mean reversion
theta = 0.1626**2  # Variance long terme
xi    = 0.3        # Vol de vol
rho   = -0.7       # Correlation prix/vol (negatif = skew baissier)

# ============================================================
# PRICING HESTON PAR INTEGRATION (formule semi-analytique)
# On utilise la fonction caracteristique du modele Heston
# ============================================================

def heston_cf(phi, S, T, r, q, v0, kappa, theta, xi, rho):
    """
    Fonction caracteristique du modele Heston
    phi : variable de Fourier
    """
    i     = complex(0, 1)
    x     = np.log(S)
    a     = kappa * theta

    # Coefficients
    d = np.sqrt((rho * xi * i * phi - kappa)**2 + xi**2 * (i * phi + phi**2))
    g = (kappa - rho * xi * i * phi - d) / (kappa - rho * xi * i * phi + d)

    # Exponentielle
    exp_dT  = np.exp(-d * T)
    C = (r - q) * i * phi * T + (a / xi**2) * (
        (kappa - rho * xi * i * phi - d) * T
        - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((kappa - rho * xi * i * phi - d) / xi**2) * (
        (1 - exp_dT) / (1 - g * exp_dT)
    )

    return np.exp(C + D * v0 + i * phi * x)

def heston_price(S, K, T, r, q, v0, kappa, theta, xi, rho, type_option='call'):
    """
    Prix d une option par le modele Heston
    Utilise l integration numerique de la fonction caracteristique
    """
    i = complex(0, 1)

    def integrand_P1(phi):
        cf    = heston_cf(phi - i, S, T, r, q, v0, kappa, theta, xi, rho)
        cf_0  = heston_cf(-i,      S, T, r, q, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-i * phi * np.log(K)) * cf / (i * phi * cf_0))

    def integrand_P2(phi):
        cf = heston_cf(phi, S, T, r, q, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-i * phi * np.log(K)) * cf / (i * phi))

    P1 = 0.5 + (1/np.pi) * quad(integrand_P1, 1e-6, 200, limit=200)[0]
    P2 = 0.5 + (1/np.pi) * quad(integrand_P2, 1e-6, 200, limit=200)[0]

    call = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

    if type_option == 'call':
        return call
    else:
        return call - S * np.exp(-q * T) + K * np.exp(-r * T)

# ============================================================
# BLACK-SCHOLES POUR COMPARAISON
# ============================================================

def bs_price(S, K, T, r, sigma, type_option='call'):
    d1  = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2  = d1 - sigma * np.sqrt(T)
    if type_option == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ============================================================
# VOL IMPLICITE BS
# On inverse BS numeriquement pour trouver sigma tel que
# BS(sigma) = Prix Heston
# Methode : bissection
# ============================================================

def vol_implicite(prix_marche, S, K, T, r, type_option='call'):
    """
    Trouve la vol implicite par bissection
    On cherche sigma tel que BS(sigma) = prix_marche
    """
    sigma_bas  = 0.001
    sigma_haut = 5.0

    for _ in range(200):
        sigma_mid = (sigma_bas + sigma_haut) / 2
        prix_mid  = bs_price(S, K, T, r, sigma_mid, type_option)

        if prix_mid < prix_marche:
            sigma_bas = sigma_mid
        else:
            sigma_haut = sigma_mid

        if abs(prix_mid - prix_marche) < 1e-8:
            break

    return sigma_mid

# ============================================================
# SMILE DE VOLATILITE
# On calcule le prix Heston pour differents strikes
# Puis on inverse BS pour obtenir la vol implicite
# Si BS etait parfait la vol implicite serait plate
# Heston genere naturellement un smile/skew
# ============================================================

strikes     = np.linspace(S * 0.80, S * 1.20, 40)
moneyness   = strikes / S  # ratio K/S

vols_impl_heston = []
vols_impl_bs     = []
prix_heston_list = []
prix_bs_list     = []

sigma_bs = np.sqrt(v0)  # vol BS = racine de la variance initiale Heston

print("Calcul du smile de volatilite en cours...")

for K_i in strikes:
    # Prix Heston
    p_heston = heston_price(S, K_i, T, r, q, v0, kappa, theta, xi, rho, 'call')
    prix_heston_list.append(p_heston)

    # Vol implicite extraite du prix Heston
    vi = vol_implicite(p_heston, S, K_i, T, r, 'call')
    vols_impl_heston.append(vi)

    # Prix BS flat (reference)
    p_bs = bs_price(S, K_i, T, r, sigma_bs, 'call')
    prix_bs_list.append(p_bs)
    vols_impl_bs.append(sigma_bs)

# ============================================================
# AFFICHAGE RESULTATS ATM
# ============================================================

K_atm        = S
prix_h_atm   = heston_price(S, K_atm, T, r, q, v0, kappa, theta, xi, rho, 'call')
prix_bs_atm  = bs_price(S, K_atm, T, r, sigma_bs, 'call')
vi_atm       = vol_implicite(prix_h_atm, S, K_atm, T, r, 'call')

print("=" * 60)
print("MODULE 6 — HESTON vs BLACK-SCHOLES")
print("=" * 60)
print(f"Parametres Heston :")
print(f"  v0    (variance initiale)    : {v0:.4f}  (sigma={np.sqrt(v0):.2%})")
print(f"  kappa (mean reversion)       : {kappa:.2f}")
print(f"  theta (variance long terme)  : {theta:.4f}  (sigma={np.sqrt(theta):.2%})")
print(f"  xi    (vol de vol)           : {xi:.2f}")
print(f"  rho   (correlation)          : {rho:.2f}")
print()
print(f"{'':30} {'Heston':>10} {'BS':>10}")
print("-" * 55)
print(f"{'Call ATM prix':30} {prix_h_atm:>10.2f} {prix_bs_atm:>10.2f}")
print(f"{'Vol implicite ATM':30} {vi_atm:>9.2%} {sigma_bs:>9.2%}")
print()
print(f"Difference de prix ATM : {prix_h_atm - prix_bs_atm:.4f}")

# ============================================================
# GRAPHIQUES
# ============================================================

plt.style.use('dark_background')
bg_color   = "#0B0E14"
cyan       = "#00F0FF"
green      = "#00FF66"
pink       = "#FF007F"
orange     = "#FF8C00"
yellow     = "#FFD700"
grid_color = "#1E2A38"

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor(bg_color)

# --- Graphique 1 : Smile de volatilite ---
ax = axes[0]
ax.set_facecolor(bg_color)
ax.plot(moneyness, [v*100 for v in vols_impl_heston],
        color=cyan,  linewidth=2.5, label='Vol implicite Heston')
ax.plot(moneyness, [v*100 for v in vols_impl_bs],
        color=green, linewidth=2, linestyle='--', label='Vol implicite BS (flat)')
ax.axvline(x=1.0, color='white', linewidth=0.8, linestyle='--', alpha=0.5, label='ATM')
ax.set_title('Smile de volatilite', color=cyan, fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Moneyness (K/S)', color='white', fontsize=10)
ax.set_ylabel('Vol implicite (%)', color='white', fontsize=10)
ax.tick_params(colors='white', labelsize=9)
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=9)

# --- Graphique 2 : Prix Heston vs BS ---
ax = axes[1]
ax.set_facecolor(bg_color)
ax.plot(moneyness, prix_heston_list,
        color=cyan,  linewidth=2.5, label='Prix Heston')
ax.plot(moneyness, prix_bs_list,
        color=green, linewidth=2, linestyle='--', label='Prix BS')
ax.axvline(x=1.0, color='white', linewidth=0.8, linestyle='--', alpha=0.5, label='ATM')
ax.set_title('Prix Call : Heston vs BS', color=cyan, fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Moneyness (K/S)', color='white', fontsize=10)
ax.set_ylabel('Prix', color='white', fontsize=10)
ax.tick_params(colors='white', labelsize=9)
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=9)

# --- Graphique 3 : Difference de prix Heston - BS ---
ax = axes[2]
ax.set_facecolor(bg_color)
diff = [h - b for h, b in zip(prix_heston_list, prix_bs_list)]
couleurs_diff = [green if d >= 0 else pink for d in diff]
ax.bar(moneyness, diff, width=0.008, color=couleurs_diff, alpha=0.8)
ax.axhline(y=0, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax.axvline(x=1.0, color='white', linewidth=0.8, linestyle='--', alpha=0.5, label='ATM')
ax.set_title('Difference prix Heston - BS', color=cyan, fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Moneyness (K/S)', color='white', fontsize=10)
ax.set_ylabel('Difference de prix', color='white', fontsize=10)
ax.tick_params(colors='white', labelsize=9)
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=9)

fig.suptitle(
    f'Heston vs BS   |   kappa={kappa}   theta={np.sqrt(theta):.2%}   xi={xi}   rho={rho}',
    color=yellow, fontsize=12, fontweight='bold'
)

plt.tight_layout()
plt.savefig('sp500_heston.png', dpi=150, bbox_inches='tight', facecolor=bg_color)
plt.show()
print("Graphique sauvegarde : sp500_heston.png")