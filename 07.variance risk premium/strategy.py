import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

df_sp500 = pd.read_csv('SP500.csv', parse_dates=['Date'])
df_sp500 = df_sp500.sort_values('Date').reset_index(drop=True)

df_vix = pd.read_csv('VIX.csv')
df_vix.columns = ['Date', 'VIX']
df_vix['Date'] = pd.to_datetime(df_vix['Date'])
df_vix = df_vix.sort_values('Date').reset_index(drop=True)
df_vix = df_vix.dropna()

# Merge sur les dates communes
df = pd.merge(df_sp500[['Date', 'Price']], df_vix, on='Date', how='inner')
df = df.sort_values('Date').reset_index(drop=True)

print(f"Donnees mergees : {len(df)} jours")
print(f"Periode : {df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}")
print(df.head(5))

# ============================================================
# RENDEMENTS LOG DAILY SP500
# ============================================================

rendements_log = []
for i in range(1, len(df)):
    r = np.log(df['Price'].iloc[i] / df['Price'].iloc[i-1])
    rendements_log.append(r)

df_r = df.iloc[1:].copy().reset_index(drop=True)
df_r['r_log'] = rendements_log

# ============================================================
# VOL IMPLICITE DAILY (VIX)
# VIX est en % annualise — on le convertit en vol daily
# sigma_impl_daily = VIX / 100 / sqrt(252)
# ============================================================

df_r['sigma_impl_daily'] = df_r['VIX'] / 100 / np.sqrt(252)
df_r['sigma_impl_annuel'] = df_r['VIX'] / 100

# ============================================================
# BACKTEST SHORT VOL MENSUEL
# Logique :
# - Debut de chaque mois on "vend" un straddle ATM
# - On encaisse la prime = valeur du straddle BS
# - A la fin du mois on paie la valeur realisee
# - P&L = prime encaissee - cout de couverture realise
#
# Approximation via variance risk premium :
# P&L mensuel = (sigma_impl² - sigma_realisee²) * T
# Positif si vol implicite > vol realisee
# ============================================================

df_r['Mois'] = df_r['Date'].dt.to_period('M')
mois_liste   = df_r['Mois'].unique()

resultats = []

for mois in mois_liste:
    df_mois = df_r[df_r['Mois'] == mois].copy()

    if len(df_mois) < 5:
        continue

    # Vol implicite au debut du mois (VIX day 1)
    sigma_impl = df_mois['sigma_impl_annuel'].iloc[0]

    # Vol realisee sur le mois
    n_jours_mois = len(df_mois)
    somme_carres = 0
    for r in df_mois['r_log']:
        somme_carres += r**2
    variance_realisee = somme_carres / n_jours_mois * 252
    sigma_realisee    = variance_realisee ** 0.5

    # T en annees
    T = n_jours_mois / 252

    # P&L variance risk premium
    # On vend la variance implicite, on achete la variance realisee
    pnl_vrp = (sigma_impl**2 - variance_realisee) * T * 100

    # Prix du sous-jacent debut de mois
    S_debut = df_mois['Price'].iloc[0]

    resultats.append({
        'Mois'            : str(mois),
        'S_debut'         : S_debut,
        'sigma_impl'      : sigma_impl,
        'sigma_realisee'  : sigma_realisee,
        'VRP'             : sigma_impl - sigma_realisee,
        'PnL_VRP'         : pnl_vrp,
        'n_jours'         : n_jours_mois,
    })

df_res = pd.DataFrame(resultats)

# ============================================================
# STATISTIQUES GLOBALES
# ============================================================

pnl_cumul       = df_res['PnL_VRP'].cumsum()
pnl_total       = df_res['PnL_VRP'].sum()
pnl_moyen       = df_res['PnL_VRP'].mean()
win_rate        = (df_res['PnL_VRP'] > 0).sum() / len(df_res)
vrp_moyen       = df_res['VRP'].mean()
vol_impl_moy    = df_res['sigma_impl'].mean()
vol_real_moy    = df_res['sigma_realisee'].mean()
mois_positifs   = (df_res['PnL_VRP'] > 0).sum()
mois_negatifs   = (df_res['PnL_VRP'] < 0).sum()

print()
print("=" * 60)
print("MODULE 7 — BACKTEST SHORT VOL SYSTEMATIQUE")
print("=" * 60)
print(f"Nombre de mois                : {len(df_res)}")
print()
print(f"Vol implicite moyenne (VIX)   : {vol_impl_moy:.2%}")
print(f"Vol realisee moyenne          : {vol_real_moy:.2%}")
print(f"Variance Risk Premium moyenne : {vrp_moyen:.2%}")
print()
print(f"P&L total (VRP cumulee)       : {pnl_total:.4f}")
print(f"P&L moyen par mois            : {pnl_moyen:.4f}")
print(f"Mois positifs / negatifs      : {mois_positifs} / {mois_negatifs}")
print(f"Win rate                      : {win_rate:.1%}")
print()
print(f"{'Mois':10} {'S':>8} {'Vol Impl':>10} {'Vol Real':>10} {'VRP':>8} {'PnL':>8}")
print("-" * 60)
for _, row in df_res.iterrows():
    print(f"{row['Mois']:10} {row['S_debut']:>8.0f} {row['sigma_impl']:>9.2%} {row['sigma_realisee']:>9.2%} {row['VRP']:>7.2%} {row['PnL_VRP']:>8.4f}")

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

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor(bg_color)

mois_labels = df_res['Mois'].values

# --- Graphique 1 : Vol implicite vs vol realisee ---
ax = axes[0, 0]
ax.set_facecolor(bg_color)
ax.plot(range(len(df_res)), df_res['sigma_impl']*100,
        color=cyan,  linewidth=2, label='Vol implicite (VIX)')
ax.plot(range(len(df_res)), df_res['sigma_realisee']*100,
        color=orange, linewidth=2, linestyle='--', label='Vol realisee')
ax.fill_between(range(len(df_res)),
                df_res['sigma_impl']*100,
                df_res['sigma_realisee']*100,
                where=df_res['sigma_impl'] > df_res['sigma_realisee'],
                color=green, alpha=0.2, label='VRP positive (short vol gagne)')
ax.fill_between(range(len(df_res)),
                df_res['sigma_impl']*100,
                df_res['sigma_realisee']*100,
                where=df_res['sigma_impl'] <= df_res['sigma_realisee'],
                color=pink, alpha=0.3, label='VRP negative (short vol perd)')
ax.set_title('Vol implicite vs Vol realisee', color=cyan, fontsize=12, fontweight='bold')
ax.set_xlabel('Mois', color='white')
ax.set_ylabel('Volatilite (%)', color='white')
ax.tick_params(colors='white', labelsize=8)
ax.set_xticks(range(0, len(df_res), 4))
ax.set_xticklabels(mois_labels[::4], rotation=45, ha='right')
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=8)

# --- Graphique 2 : P&L mensuel ---
ax = axes[0, 1]
ax.set_facecolor(bg_color)
couleurs_pnl = [green if p > 0 else pink for p in df_res['PnL_VRP']]
ax.bar(range(len(df_res)), df_res['PnL_VRP'], color=couleurs_pnl, alpha=0.8)
ax.axhline(y=0, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax.set_title('P&L mensuel short vol', color=cyan, fontsize=12, fontweight='bold')
ax.set_xlabel('Mois', color='white')
ax.set_ylabel('P&L (VRP)', color='white')
ax.tick_params(colors='white', labelsize=8)
ax.set_xticks(range(0, len(df_res), 4))
ax.set_xticklabels(mois_labels[::4], rotation=45, ha='right')
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)

# --- Graphique 3 : P&L cumule ---
ax = axes[1, 0]
ax.set_facecolor(bg_color)
ax.plot(range(len(df_res)), pnl_cumul,
        color=cyan, linewidth=2.5, label='P&L cumule')
ax.fill_between(range(len(df_res)), pnl_cumul, 0,
                where=pnl_cumul >= 0, color=green, alpha=0.15)
ax.fill_between(range(len(df_res)), pnl_cumul, 0,
                where=pnl_cumul < 0, color=pink, alpha=0.3)
ax.axhline(y=0, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax.set_title('P&L cumule short vol', color=cyan, fontsize=12, fontweight='bold')
ax.set_xlabel('Mois', color='white')
ax.set_ylabel('P&L cumule', color='white')
ax.tick_params(colors='white', labelsize=8)
ax.set_xticks(range(0, len(df_res), 4))
ax.set_xticklabels(mois_labels[::4], rotation=45, ha='right')
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=8)

# --- Graphique 4 : Distribution VRP ---
ax = axes[1, 1]
ax.set_facecolor(bg_color)
vrp_vals = df_res['VRP'].values * 100
couleurs_hist = [green if v > 0 else pink for v in vrp_vals]
ax.hist(vrp_vals, bins=20, color=cyan, alpha=0.5, density=True, label='Distribution VRP')
ax.axvline(x=0,           color='white', linewidth=1,   linestyle='--', alpha=0.5)
ax.axvline(x=vrp_moyen*100, color=yellow, linewidth=1.5, linestyle='--',
           label=f'Moyenne {vrp_moyen*100:.1f}%')
ax.set_title('Distribution Variance Risk Premium', color=cyan, fontsize=12, fontweight='bold')
ax.set_xlabel('VRP = Vol impl - Vol realisee (%)', color='white')
ax.set_ylabel('Densite', color='white')
ax.tick_params(colors='white', labelsize=8)
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=8)

fig.suptitle(
    f'Short Vol Systematique   |   VRP moyenne: {vrp_moyen:.2%}   Win rate: {win_rate:.1%}   Mois+: {mois_positifs}   Mois-: {mois_negatifs}',
    color=yellow, fontsize=12, fontweight='bold'
)

plt.tight_layout()
plt.savefig('sp500_short_vol.png', dpi=150, bbox_inches='tight', facecolor=bg_color)
plt.show()
print("Graphique sauvegarde : sp500_short_vol.png")