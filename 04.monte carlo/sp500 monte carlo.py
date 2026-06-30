import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('SP500.csv', parse_dates=['Date'])

# ============================================================
# RENDEMENTS ET CALIBRATION
# ============================================================

rendements_log = []
for i in range(1, len(df)):
    P_aujourd_hui = df['Price'].iloc[i]
    P_hier        = df['Price'].iloc[i - 1]
    r             = np.log(P_aujourd_hui / P_hier)
    rendements_log.append(r)

df_r = df.iloc[1:].copy()
df_r['r_log'] = rendements_log

n_jours  = len(df_r)
n_annees = n_jours / 252

somme_log = 0
for r in df_r['r_log']:
    somme_log += r
mu_daily = somme_log / n_jours

somme_carres = 0
for r in df_r['r_log']:
    somme_carres += (r - mu_daily) ** 2
variance_daily = somme_carres / (n_jours - 1)
sigma_daily    = variance_daily ** 0.5

mu_annualise    = mu_daily * 252
sigma_annualise = sigma_daily * np.sqrt(252)

print("=" * 55)
print("MODULE 4 — MONTE CARLO GBM")
print("=" * 55)
print(f"Mu annualise    : {mu_annualise:.2%}")
print(f"Sigma annualise : {sigma_annualise:.2%}")
print(f"Prix initial    : {df['Price'].iloc[0]:.2f}")
print(f"Nb jours        : {n_jours}")
print()

# ============================================================
# SIMULATION MONTE CARLO
# S(t+1) = S(t) * exp((mu - sigma²/2) * dt + sigma * sqrt(dt) * Z)
# ============================================================

np.random.seed(42)  # reproductibilite

n_simulations = 1000
dt            = 1 / 252
S0            = df['Price'].iloc[0]

# Matrice de trajectoires : n_jours lignes x n_simulations colonnes
trajectoires = np.zeros((n_jours + 1, n_simulations))
trajectoires[0] = S0

for t in range(1, n_jours + 1):
    # Tirer n_simulations chocs gaussiens independants
    Z = np.random.standard_normal(n_simulations)
    # Appliquer la formule GBM
    trajectoires[t] = trajectoires[t-1] * np.exp(
        (mu_annualise - 0.5 * sigma_annualise**2) * dt
        + sigma_annualise * np.sqrt(dt) * Z
    )

# Prix finaux apres n_jours
prix_finaux = trajectoires[-1]

# Statistiques sur les prix finaux
prix_final_median  = np.median(prix_finaux)
prix_final_moyen   = np.mean(prix_finaux)
prix_final_reel    = df['Price'].iloc[-1]
percentile_5       = np.percentile(prix_finaux, 5)
percentile_95      = np.percentile(prix_finaux, 95)

# CAGR median simule
cagr_median_simule = (prix_final_median / S0) ** (1 / n_annees) - 1
cagr_reel          = (prix_final_reel   / S0) ** (1 / n_annees) - 1

print(f"Nb simulations         : {n_simulations}")
print(f"Prix final reel        : {prix_final_reel:.2f}")
print(f"Prix final moyen sim.  : {prix_final_moyen:.2f}")
print(f"Prix final median sim. : {prix_final_median:.2f}")
print(f"Percentile 5%          : {percentile_5:.2f}")
print(f"Percentile 95%         : {percentile_95:.2f}")
print()
print(f"CAGR reel              : {cagr_reel:.2%}")
print(f"CAGR median simule     : {cagr_median_simule:.2%}")

# ============================================================
# GRAPHIQUE
# ============================================================

plt.style.use('dark_background')
bg_color   = "#0B0E14"
cyan       = "#00F0FF"
green      = "#00FF66"
pink       = "#FF007F"
orange     = "#FF8C00"
yellow     = "#FFD700"
grid_color = "#1E2A38"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(bg_color)

# --- Graphique 1 : Trajectoires simulees ---
ax1.set_facecolor(bg_color)

# Tracer 200 trajectoires en transparence
for i in range(200):
    ax1.plot(trajectoires[:, i], color=cyan, alpha=0.03, linewidth=0.5)

# Bande percentile 5-95
ax1.fill_between(
    range(n_jours + 1),
    np.percentile(trajectoires, 5,  axis=1),
    np.percentile(trajectoires, 95, axis=1),
    color=cyan, alpha=0.15, label='Intervalle 5%-95%'
)

# Trajectoire mediane
ax1.plot(np.median(trajectoires, axis=1), color=green,  linewidth=2,   label='Mediane simulee')

# Trajectoire reelle
ax1.plot(df['Price'].values, color=orange, linewidth=2, label='Trajectoire reelle')

ax1.set_title('Monte Carlo GBM — 1000 trajectoires',
              color=cyan, fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Jours', color='white', fontsize=10)
ax1.set_ylabel('Prix', color='white', fontsize=10)
ax1.tick_params(colors='white', labelsize=9)
ax1.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax1.set_axisbelow(True)
for spine in ax1.spines.values():
    spine.set_edgecolor(grid_color)
ax1.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=9)

# --- Graphique 2 : Distribution des prix finaux ---
ax2.set_facecolor(bg_color)

ax2.hist(prix_finaux, bins=60, color=cyan, alpha=0.5, density=True, label='Prix finaux simules')
ax2.axvline(x=prix_final_reel,   color=orange, linewidth=2,   linestyle='--', label=f'Prix reel {prix_final_reel:.0f}')
ax2.axvline(x=prix_final_median, color=green,  linewidth=2,   linestyle='--', label=f'Mediane {prix_final_median:.0f}')
ax2.axvline(x=percentile_5,      color=pink,   linewidth=1.5, linestyle=':',  label=f'P5 {percentile_5:.0f}')
ax2.axvline(x=percentile_95,     color=yellow, linewidth=1.5, linestyle=':',  label=f'P95 {percentile_95:.0f}')

ax2.set_title('Distribution des prix finaux — Monte Carlo',
              color=cyan, fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Prix final', color='white', fontsize=10)
ax2.set_ylabel('Densite', color='white', fontsize=10)
ax2.tick_params(colors='white', labelsize=9)
ax2.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax2.set_axisbelow(True)
for spine in ax2.spines.values():
    spine.set_edgecolor(grid_color)
ax2.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=9)

fig.suptitle(
    f'Monte Carlo GBM   |   Mu: {mu_annualise:.2%}   Sigma: {sigma_annualise:.2%}   N simulations: {n_simulations}',
    color=yellow, fontsize=12, fontweight='bold'
)

plt.tight_layout()
plt.savefig('sp500_monte_carlo.png', dpi=150, bbox_inches='tight', facecolor=bg_color)
plt.show()
print("Graphique sauvegarde : sp500_monte_carlo.png")