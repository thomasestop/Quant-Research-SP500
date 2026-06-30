import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# ============================================================
# CAGR ET RENDEMENT ANNUALISE
# ============================================================

prix_initial = df['Price'].iloc[0]
prix_final   = df['Price'].iloc[-1]
cagr         = (prix_final / prix_initial) ** (1 / n_annees) - 1

# ============================================================
# DRAWDOWN
# Le drawdown mesure la chute depuis le plus haut historique
# Drawdown(t) = (Prix(t) - Max(Prix de 0 a t)) / Max(Prix de 0 a t)
# ============================================================

prix = df['Price'].values
dates = df['Date'].values

drawdowns     = []
plus_haut     = prix[0]

for p in prix:
    if p > plus_haut:
        plus_haut = p
    dd = (p - plus_haut) / plus_haut
    drawdowns.append(dd)

df['drawdown'] = drawdowns

# Maximum drawdown
max_drawdown = min(drawdowns)
date_max_dd  = df['Date'].iloc[drawdowns.index(max_drawdown)]

# Duree du drawdown maximum
# On cherche quand le drawdown a commence et quand il s'est termine
indice_bottom = drawdowns.index(max_drawdown)

# Chercher le debut du drawdown (dernier plus haut avant le bottom)
indice_debut = indice_bottom
for i in range(indice_bottom, -1, -1):
    if drawdowns[i] == 0:
        indice_debut = i
        break

# Chercher la fin du drawdown (prochain retour a zero apres le bottom)
indice_fin = indice_bottom
for i in range(indice_bottom, len(drawdowns)):
    if drawdowns[i] >= 0:
        indice_fin = i
        break

duree_drawdown = indice_fin - indice_debut

# ============================================================
# RATIOS DE PERFORMANCE
# ============================================================

taux_sans_risque = 0.04  # 4% annuel 
rf_daily         = taux_sans_risque / 252

# --- Sharpe ---
# Mesure le rendement en exces par unite de volatilite totale
# Sharpe = (mu_annualise - rf) / sigma_annualise

excess_return_annualise = mu_daily * 252 - taux_sans_risque
sharpe = excess_return_annualise / (sigma_daily * np.sqrt(252))

# --- Sortino ---
# Comme Sharpe mais on ne penalise que la volatilite negative
# On calcule le downside deviation = volatilite des rendements negatifs seulement

rendements_negatifs = []
for r in df_r['r_log']:
    if r < rf_daily:
        rendements_negatifs.append((r - rf_daily) ** 2)
    else:
        rendements_negatifs.append(0)

downside_variance = sum(rendements_negatifs) / (n_jours - 1)
downside_deviation = downside_variance ** 0.5 * np.sqrt(252)

sortino = excess_return_annualise / downside_deviation

# --- Calmar ---
# Mesure le rendement par rapport au drawdown maximum
# Calmar = CAGR / |Max Drawdown|

calmar = cagr / abs(max_drawdown)

# ============================================================
# AFFICHAGE
# ============================================================

print("=" * 55)
print("MODULE 2 — DRAWDOWN ET RATIOS DE PERFORMANCE")
print("=" * 55)
print()
print("--- DRAWDOWN ---")
print(f"Maximum Drawdown         : {max_drawdown:.2%}")
print(f"Date du plus bas         : {date_max_dd.date()}")
print(f"Duree du drawdown        : {duree_drawdown} jours de trading")
print(f"                           ({duree_drawdown/252*12:.1f} mois)")
print()
print("--- RATIOS ---")
print(f"Taux sans risque         : {taux_sans_risque:.2%}")
print(f"CAGR                     : {cagr:.2%}")
print(f"Sigma annualise          : {sigma_daily*np.sqrt(252):.2%}")
print(f"Downside deviation       : {downside_deviation:.2%}")
print()
print(f"Sharpe ratio             : {sharpe:.3f}")
print(f"Sortino ratio            : {sortino:.3f}")
print(f"Calmar ratio             : {calmar:.3f}")
print()
print("Interpretation :")
print(f"  Sharpe  {sharpe:.2f} -> {'bon' if sharpe > 1 else 'acceptable' if sharpe > 0.5 else 'faible'} (>1 = excellent, >0.5 = acceptable)")
print(f"  Sortino {sortino:.2f} -> penalise uniquement la vol negative")
print(f"  Calmar  {calmar:.2f} -> tu gagnes {calmar:.2f}x ton pire drawdown par an")

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
grid_color = "#1E2A38"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig.patch.set_facecolor(bg_color)

# --- Graphique 1 : Prix ---
ax1.set_facecolor(bg_color)
ax1.plot(df['Date'], df['Price'], color=cyan, linewidth=1.5)
ax1.fill_between(df['Date'], df['Price'], alpha=0.08, color=cyan)
ax1.axvline(x=date_max_dd, color=pink, linewidth=1.2, linestyle='--', alpha=0.8, label=f'Max DD ({max_drawdown:.2%})')
ax1.set_title('SP500 — Prix et Drawdown', color=cyan, fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('Prix', color='white', fontsize=10)
ax1.tick_params(colors='white', labelsize=9)
ax1.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax1.set_axisbelow(True)
for spine in ax1.spines.values():
    spine.set_edgecolor(grid_color)
ax1.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=9)

# --- Graphique 2 : Drawdown ---
ax2.set_facecolor(bg_color)
ax2.fill_between(df['Date'], df['drawdown'], 0, color=pink, alpha=0.5, label='Drawdown')
ax2.plot(df['Date'], df['drawdown'], color=pink, linewidth=0.8)
ax2.axhline(y=max_drawdown, color=orange, linewidth=1.2, linestyle='--', label=f'Max drawdown {max_drawdown:.2%}')
ax2.axhline(y=0, color='white', linewidth=0.5, alpha=0.4)
ax2.set_ylabel('Drawdown', color='white', fontsize=10)
ax2.set_xlabel('Date', color='white', fontsize=10)
ax2.tick_params(colors='white', labelsize=9)
ax2.xaxis.set_tick_params(rotation=30)
ax2.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax2.set_axisbelow(True)
for spine in ax2.spines.values():
    spine.set_edgecolor(grid_color)
ax2.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=9)

fig.suptitle(
    f'Sharpe: {sharpe:.2f}   Sortino: {sortino:.2f}   Calmar: {calmar:.2f}   Max DD: {max_drawdown:.2%}',
    color=yellow, fontsize=12, fontweight='bold'
)

plt.tight_layout()
plt.savefig('sp500_drawdown_ratios.png', dpi=150, bbox_inches='tight', facecolor=bg_color)
plt.show()
print("Graphique sauvegarde : sp500_drawdown_ratios.png")