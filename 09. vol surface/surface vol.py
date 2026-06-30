import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# CHARGEMENT
# ============================================================

options = pd.read_csv('spx_options.csv', parse_dates=['Expiration Date'])

S0 = 7440.4302  # spot SP500 au moment de la cotation

# ============================================================
# ON GARDE UNIQUEMENT LES CALLS OTM ET PUTS OTM
# C'est la convention standard pour construire un smile propre
# (les options OTM sont plus liquides et moins bruitees que les ITM)
# Calls OTM : strike > spot
# Puts OTM  : strike < spot
# ============================================================

calls_otm = options[(options['type']=='call') & (options['Strike'] >= S0)]
puts_otm  = options[(options['type']=='put')  & (options['Strike'] <  S0)]

smile_data = pd.concat([calls_otm, puts_otm], ignore_index=True)
smile_data = smile_data[smile_data['Expiration Date'] >= pd.Timestamp('2026-07-05')]

# On choisit quelques maturites representatives pour la lisibilite
maturites_choisies = sorted(smile_data['Expiration Date'].unique())
# Garder environ 6 maturites bien espacees
indices = np.linspace(0, len(maturites_choisies)-1, 6).astype(int)
maturites_affichees = [maturites_choisies[i] for i in indices]

# ============================================================
# GRAPHIQUE STYLE NEON
# ============================================================

plt.style.use('dark_background')
bg_color   = "#0B0E14"
cyan       = "#00F0FF"
green      = "#00FF66"
pink       = "#FF007F"
orange     = "#FF8C00"
yellow     = "#FFD700"
grid_color = "#1E2A38"
couleurs   = [cyan, green, orange, pink, yellow, "#9D4EDD"]

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor(bg_color)

# --- Graphique 1 : Smile par maturite (2D) ---
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor(bg_color)

for i, mat in enumerate(maturites_affichees):
    df_mat = smile_data[smile_data['Expiration Date'] == mat].sort_values('Strike')
    moneyness = df_mat['Strike'] / S0
    label = mat.strftime('%Y-%m-%d')
    ax1.plot(moneyness, df_mat['iv']*100, color=couleurs[i], linewidth=2, marker='o', markersize=2, label=label)

ax1.axvline(x=1.0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
ax1.set_title('Smile de volatilite par maturite', color=cyan, fontsize=13, fontweight='bold')
ax1.set_xlabel('Moneyness (K/S)', color='white')
ax1.set_ylabel('Vol implicite (%)', color='white')
ax1.tick_params(colors='white', labelsize=8)
ax1.yaxis.grid(True, color=grid_color, linewidth=0.5)
for spine in ax1.spines.values():
    spine.set_edgecolor(grid_color)
ax1.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=7)

# --- Graphique 2 : Term structure ATM ---
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor(bg_color)

term_structure = []
for mat in maturites_choisies:
    df_mat = smile_data[smile_data['Expiration Date'] == mat].copy()
    df_mat['dist_atm'] = abs(df_mat['Strike'] - S0)
    plus_proche = df_mat.nsmallest(1, 'dist_atm')
    if len(plus_proche) > 0:
        T_jours = (mat - pd.Timestamp('2026-06-30')).days
        term_structure.append({'T_jours': T_jours, 'iv_atm': plus_proche['iv'].values[0]})

df_term = pd.DataFrame(term_structure).sort_values('T_jours')

ax2.plot(df_term['T_jours'], df_term['iv_atm']*100, color=cyan, linewidth=2, marker='o', markersize=4)
ax2.set_title('Term Structure — Vol implicite ATM', color=cyan, fontsize=13, fontweight='bold')
ax2.set_xlabel('Jours jusqu a maturite', color='white')
ax2.set_ylabel('Vol implicite ATM (%)', color='white')
ax2.tick_params(colors='white', labelsize=8)
ax2.yaxis.grid(True, color=grid_color, linewidth=0.5)
for spine in ax2.spines.values():
    spine.set_edgecolor(grid_color)

# --- Graphique 3 : Surface 3D ---
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.set_facecolor(bg_color)

# Construire la grille pour la surface 3D
strikes_grid = np.linspace(S0*0.8, S0*1.2, 30)
maturites_jours = sorted(smile_data['Expiration Date'].unique())
maturites_jours_num = [(m - pd.Timestamp('2026-06-30')).days for m in maturites_jours]

Z = np.full((len(maturites_jours), len(strikes_grid)), np.nan)

for i, mat in enumerate(maturites_jours):
    df_mat = smile_data[smile_data['Expiration Date'] == mat].sort_values('Strike')
    if len(df_mat) < 3:
        continue
    Z[i] = np.interp(strikes_grid, df_mat['Strike'], df_mat['iv']*100,
                       left=np.nan, right=np.nan)

X, Y = np.meshgrid(strikes_grid/S0, maturites_jours_num)

surf = ax3.plot_surface(X, Y, Z, cmap='cool', alpha=0.85, edgecolor='none')
ax3.set_title('Surface de volatilite implicite', color=cyan, fontsize=13, fontweight='bold')
ax3.set_xlabel('Moneyness (K/S)', color='white', fontsize=8)
ax3.set_ylabel('Jours maturite', color='white', fontsize=8)
ax3.set_zlabel('Vol implicite (%)', color='white', fontsize=8)
ax3.tick_params(colors='white', labelsize=7)
ax3.xaxis.pane.set_facecolor(bg_color)
ax3.yaxis.pane.set_facecolor(bg_color)
ax3.zaxis.pane.set_facecolor(bg_color)

# --- Graphique 4 : Skew (pente du smile) par maturite ---
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor(bg_color)

skews = []
for mat in maturites_choisies:
    df_mat = smile_data[smile_data['Expiration Date'] == mat].copy()
    df_mat['moneyness'] = df_mat['Strike'] / S0

    # Put 90% moneyness vs call 110% moneyness
    put_90  = df_mat[(df_mat['type']=='put')  & (df_mat['moneyness'].between(0.88,0.92))]
    call_110 = df_mat[(df_mat['type']=='call') & (df_mat['moneyness'].between(1.08,1.12))]

    if len(put_90) > 0 and len(call_110) > 0:
        T_jours = (mat - pd.Timestamp('2026-06-30')).days
        skew = put_90['iv'].mean() - call_110['iv'].mean()
        skews.append({'T_jours': T_jours, 'skew': skew})

df_skew = pd.DataFrame(skews).sort_values('T_jours')

ax4.plot(df_skew['T_jours'], df_skew['skew']*100, color=pink, linewidth=2, marker='o', markersize=4)
ax4.axhline(y=0, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax4.set_title('Skew (Put 90% - Call 110%) par maturite', color=cyan, fontsize=13, fontweight='bold')
ax4.set_xlabel('Jours jusqu a maturite', color='white')
ax4.set_ylabel('Skew (points de vol %)', color='white')
ax4.tick_params(colors='white', labelsize=8)
ax4.yaxis.grid(True, color=grid_color, linewidth=0.5)
for spine in ax4.spines.values():
    spine.set_edgecolor(grid_color)

fig.suptitle(
    f'Surface de Volatilite Implicite SPX   |   Spot: {S0:.0f}   {len(maturites_choisies)} maturites   {len(smile_data)} options',
    color=yellow, fontsize=13, fontweight='bold'
)

plt.tight_layout()
plt.savefig('spx_vol_surface.png', dpi=150, bbox_inches='tight', facecolor=bg_color)
plt.show()
print("Graphique sauvegarde : spx_vol_surface.png")