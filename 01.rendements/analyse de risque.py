import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

df = pd.read_csv('SP500.csv', parse_dates=['Date'])

# ============================================================
# RENDEMENTS
# ============================================================

rendements_log = []
for i in range(1, len(df)):
    P_aujourd_hui = df['Price'].iloc[i]
    P_hier        = df['Price'].iloc[i - 1]
    r             = np.log(P_aujourd_hui / P_hier)
    rendements_log.append(r)

df_r = df.iloc[1:].copy()
df_r['r_log'] = rendements_log

n_jours = len(df_r)

somme_log = 0
for r in df_r['r_log']:
    somme_log += r
mu_daily = somme_log / n_jours

somme_carres = 0
for r in df_r['r_log']:
    somme_carres += (r - mu_daily) ** 2
variance_daily = somme_carres / (n_jours - 1)
sigma_daily    = variance_daily ** 0.5

# ============================================================
# VaR HISTORIQUE
# Trier les rendements et prendre le percentile 5%
# ============================================================

rendements_tries = sorted(df_r['r_log'].values)

# Indice du 5eme percentile
indice_95 = int(n_jours * 0.05)
indice_99 = int(n_jours * 0.01)

var_hist_95 = rendements_tries[indice_95]
var_hist_99 = rendements_tries[indice_99]

# ============================================================
# VaR PARAMETRIQUE
# On suppose gaussienne : VaR = mu - z * sigma
# z = 1.645 pour 95%, z = 2.326 pour 99%
# ============================================================

z_95 = 1.645
z_99 = 2.326

var_param_95 = mu_daily - z_95 * sigma_daily
var_param_99 = mu_daily - z_99 * sigma_daily

# ============================================================
# CVaR (Expected Shortfall)
# Moyenne des rendements qui depassent la VaR
# C'est la perte moyenne dans les pires cas
# ============================================================

# CVaR historique 95%
pertes_au_dela_95 = []
for r in rendements_tries:
    if r <= var_hist_95:
        pertes_au_dela_95.append(r)

cvar_hist_95 = sum(pertes_au_dela_95) / len(pertes_au_dela_95)

# CVaR historique 99%
pertes_au_dela_99 = []
for r in rendements_tries:
    if r <= var_hist_99:
        pertes_au_dela_99.append(r)

cvar_hist_99 = sum(pertes_au_dela_99) / len(pertes_au_dela_99)

# CVaR parametrique
# Formule analytique : mu - sigma * phi(z) / (1 - niveau)
# phi = densite gaussienne standard au point z
cvar_param_95 = mu_daily - sigma_daily * norm.pdf(z_95) / 0.05
cvar_param_99 = mu_daily - sigma_daily * norm.pdf(z_99) / 0.01

# ============================================================
# CAPITAL A RISQUE
# Sur un portefeuille de 100 000 euros
# ============================================================

capital = 100000

print("=" * 60)
print("MODULE 2 — VaR ET CVaR")
print("=" * 60)
print(f"Capital investi        : {capital:,.0f} euros")
print(f"Nombre de jours        : {n_jours}")
print()
print(f"{'':30} {'95%':>10} {'99%':>10}")
print("-" * 55)
print(f"{'VaR historique (%)':30} {var_hist_95:>9.2%} {var_hist_99:>9.2%}")
print(f"{'VaR parametrique (%)':30} {var_param_95:>9.2%} {var_param_99:>9.2%}")
print(f"{'CVaR historique (%)':30} {cvar_hist_95:>9.2%} {cvar_hist_99:>9.2%}")
print(f"{'CVaR parametrique (%)':30} {cvar_param_95:>9.2%} {cvar_param_99:>9.2%}")
print("-" * 55)
print(f"{'VaR historique (euros)':30} {var_hist_95*capital:>9,.0f} {var_hist_99*capital:>9,.0f}")
print(f"{'CVaR historique (euros)':30} {cvar_hist_95*capital:>9,.0f} {cvar_hist_99*capital:>9,.0f}")
print("=" * 60)
print()
print("Interpretation :")
print(f"  VaR 95% : dans 95% des jours tu perds moins de {abs(var_hist_95)*100:.2f}%")
print(f"            soit moins de {abs(var_hist_95)*capital:,.0f} euros sur {capital:,} euros")
print(f"  CVaR 95%: dans les 5% pires jours tu perds en moyenne {abs(cvar_hist_95)*100:.2f}%")
print(f"            soit en moyenne {abs(cvar_hist_95)*capital:,.0f} euros")

# ============================================================
# GRAPHIQUE
# ============================================================

plt.style.use('dark_background')
bg_color = "#0B0E14"
cyan     = "#00F0FF"
green    = "#00FF66"
pink     = "#FF007F"
orange   = "#FF8C00"
yellow   = "#FFD700"

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

x = np.linspace(min(rendements_tries), max(rendements_tries), 400)
gauss = norm.pdf(x, mu_daily, sigma_daily)

ax.hist(rendements_tries, bins=60, density=True, color=cyan, alpha=0.4, label='SP500 observe')
ax.plot(x, gauss, color=green, linewidth=2, linestyle='--', label='Gaussienne theorique')

# Zone CVaR 95% historique
x_cvar = [r for r in x if r <= var_hist_95]
ax.fill_between(x_cvar, norm.pdf(x_cvar, mu_daily, sigma_daily), color=pink, alpha=0.4, label='Zone CVaR 95%')

# Lignes VaR
ax.axvline(x=var_hist_95,  color=orange, linewidth=1.5, linestyle='--', label=f'VaR hist 95% ({var_hist_95:.2%})')
ax.axvline(x=var_hist_99,  color=pink,   linewidth=1.5, linestyle='--', label=f'VaR hist 99% ({var_hist_99:.2%})')
ax.axvline(x=var_param_95, color=yellow, linewidth=1.5, linestyle=':',  label=f'VaR param 95% ({var_param_95:.2%})')

ax.set_title('VaR et CVaR — Distribution des rendements log SP500',
             color=cyan, fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Log-return', color='white', fontsize=11)
ax.set_ylabel('Densite', color='white', fontsize=11)
ax.tick_params(colors='white', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("#1E2A38")
ax.yaxis.grid(True, color="#1E2A38", linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(facecolor=bg_color, edgecolor="#1E2A38", labelcolor='white', fontsize=9)

plt.tight_layout()
plt.savefig('sp500_var_cvar.png', dpi=150, bbox_inches='tight', facecolor=bg_color)
plt.show()
print("Graphique sauvegarde : sp500_var_cvar.png")