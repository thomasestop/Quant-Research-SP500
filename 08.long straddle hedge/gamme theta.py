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

df = pd.merge(df_sp500[['Date', 'Price']], df_vix, on='Date', how='inner')
df = df.sort_values('Date').reset_index(drop=True)

print(f"Donnees mergees : {len(df)} jours")

# ============================================================
# BLACK-SCHOLES — PRIX ET GREEKS
# ============================================================

def bs_price_greeks(S, K, T, r, sigma, type_option='call'):
    """
    Retourne prix, delta, gamma, theta (par jour) pour une option BS
    """
    if T <= 0:
        # Option a maturite : valeur intrinseque
        if type_option == 'call':
            return max(S - K, 0), (1.0 if S > K else 0.0), 0.0, 0.0
        else:
            return max(K - S, 0), (-1.0 if S < K else 0.0), 0.0, 0.0

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if type_option == 'call':
        prix  = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (- S*norm.pdf(d1)*sigma/(2*np.sqrt(T))
                 - r*K*np.exp(-r*T)*norm.cdf(d2)) / 252
    else:
        prix  = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (- S*norm.pdf(d1)*sigma/(2*np.sqrt(T))
                 + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 252

    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))

    return prix, delta, gamma, theta

# ============================================================
# PARAMETRES DE LA SIMULATION
# Strategie : LONG STRADDLE ATM, 1 mois de maturite, delta-hedge daily
# On reconduit un nouveau straddle chaque mois
# ============================================================

r = 0.04
maturite_jours = 21  # environ 1 mois de trading

df['Mois'] = df['Date'].dt.to_period('M')
mois_liste = df['Mois'].unique()

# ============================================================
# BOUCLE PRINCIPALE — UN STRADDLE PAR MOIS, HEDGE QUOTIDIEN
# ============================================================

resultats_quotidiens = []

for mois in mois_liste:

    df_mois = df[df['Mois'] == mois].copy().reset_index(drop=True)

    if len(df_mois) < 5:
        continue

    # --- Setup du straddle ---
    S0          = df_mois['Price'].iloc[0]
    K           = S0  # ATM
    sigma_impl  = df_mois['VIX'].iloc[0] / 100
    n_jours     = len(df_mois)

    # Prix initial du straddle (call + put) et Greeks
    T0 = n_jours / 252
    prix_call0, delta_call0, gamma_call0, theta_call0 = bs_price_greeks(S0, K, T0, r, sigma_impl, 'call')
    prix_put0,  delta_put0,  gamma_put0,  theta_put0  = bs_price_greeks(S0, K, T0, r, sigma_impl, 'put')

    prix_straddle0 = prix_call0 + prix_put0
    delta_straddle0 = delta_call0 + delta_put0

    # On est LONG le straddle -> on l'achete -> cash sort de notre poche
    # Position en sous-jacent pour delta-hedger : on shorte delta_straddle0 unites
    position_sous_jacent = -delta_straddle0

    # Cash : on paie le straddle, on encaisse/paie le hedge initial
    cash = -prix_straddle0 - position_sous_jacent * S0

    valeur_straddle_precedente = prix_straddle0
    pnl_gamma_cumule = 0
    pnl_theta_cumule = 0

    for j in range(n_jours):
        S_t = df_mois['Price'].iloc[j]
        T_t = (n_jours - j) / 252
        T_t = max(T_t, 1/252)  # eviter division par zero

        # Re-pricer le straddle au jour j avec la meme vol implicite initiale
        # (hypothese : vol implicite fixee au moment de l'achat)
        prix_call_t, delta_call_t, gamma_call_t, theta_call_t = bs_price_greeks(S_t, K, T_t, r, sigma_impl, 'call')
        prix_put_t,  delta_put_t,  gamma_put_t,  theta_put_t  = bs_price_greeks(S_t, K, T_t, r, sigma_impl, 'put')

        prix_straddle_t  = prix_call_t + prix_put_t
        delta_straddle_t = delta_call_t + delta_put_t
        gamma_straddle_t = gamma_call_t + gamma_put_t
        theta_straddle_t = theta_call_t + theta_put_t

        # Variation de prix du sous-jacent depuis hier
        if j == 0:
            dS = 0
        else:
            S_hier = df_mois['Price'].iloc[j-1]
            dS = S_t - S_hier

        # P&L du jour sur la position de hedge (avant rebalancement)
        pnl_hedge_jour = position_sous_jacent * dS

        # P&L de la variation de valeur du straddle
        pnl_straddle_jour = prix_straddle_t - valeur_straddle_precedente

        # P&L total du jour = variation straddle + P&L hedge
        pnl_jour = pnl_straddle_jour + pnl_hedge_jour

        # Decomposition approximative gamma vs theta
        # gamma P&L = 0.5 * gamma * dS^2
        # theta P&L = theta (deja par jour)
        pnl_gamma_jour = 0.5 * gamma_straddle_t * dS**2
        pnl_theta_jour = theta_straddle_t

        pnl_gamma_cumule += pnl_gamma_jour
        pnl_theta_cumule += pnl_theta_jour

        resultats_quotidiens.append({
            'Date'              : df_mois['Date'].iloc[j],
            'Mois'              : str(mois),
            'S'                 : S_t,
            'K'                 : K,
            'sigma_impl'        : sigma_impl,
            'prix_straddle'     : prix_straddle_t,
            'delta_straddle'    : delta_straddle_t,
            'gamma_straddle'    : gamma_straddle_t,
            'theta_straddle'    : theta_straddle_t,
            'dS'                : dS,
            'pnl_jour'          : pnl_jour,
            'pnl_gamma_jour'    : pnl_gamma_jour,
            'pnl_theta_jour'    : pnl_theta_jour,
        })

        # Rebalancer le hedge : nouvelle position = -delta_straddle_t
        rebalancement = -delta_straddle_t - position_sous_jacent
        cash -= rebalancement * S_t
        position_sous_jacent = -delta_straddle_t

        valeur_straddle_precedente = prix_straddle_t

df_sim = pd.DataFrame(resultats_quotidiens)

# ============================================================
# STATISTIQUES GLOBALES
# ============================================================

pnl_total       = df_sim['pnl_jour'].sum()
pnl_gamma_total = df_sim['pnl_gamma_jour'].sum()
pnl_theta_total = df_sim['pnl_theta_jour'].sum()
jours_positifs  = (df_sim['pnl_jour'] > 0).sum()
jours_negatifs  = (df_sim['pnl_jour'] < 0).sum()

df_sim['pnl_cumule'] = df_sim['pnl_jour'].cumsum()

print("=" * 60)
print("SIMULATION LONG STRADDLE DELTA-HEDGE")
print("=" * 60)
print(f"Nombre de jours simules    : {len(df_sim)}")
print(f"P&L total                  : {pnl_total:.2f}")
print(f"P&L gamma cumule           : {pnl_gamma_total:.2f}")
print(f"P&L theta cumule           : {pnl_theta_total:.2f}")
print(f"Jours positifs / negatifs  : {jours_positifs} / {jours_negatifs}")
print()
print("Interpretation :")
print(f"  Le gamma a rapporte {pnl_gamma_total:.2f} (gains sur les mouvements de prix)")
print(f"  Le theta a coute {pnl_theta_total:.2f} (cout du temps qui passe)")
print(f"  Net : {pnl_gamma_total + pnl_theta_total:.2f}")

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

# --- Graphique 1 : P&L cumule total ---
ax = axes[0, 0]
ax.set_facecolor(bg_color)
ax.plot(df_sim['Date'], df_sim['pnl_cumule'], color=cyan, linewidth=2)
ax.fill_between(df_sim['Date'], df_sim['pnl_cumule'], 0,
                where=df_sim['pnl_cumule']>=0, color=green, alpha=0.15)
ax.fill_between(df_sim['Date'], df_sim['pnl_cumule'], 0,
                where=df_sim['pnl_cumule']<0, color=pink, alpha=0.3)
ax.axhline(y=0, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax.set_title('P&L cumule — Long Straddle Delta-Hedge', color=cyan, fontsize=12, fontweight='bold')
ax.set_ylabel('P&L', color='white')
ax.tick_params(colors='white', labelsize=8)
ax.xaxis.set_tick_params(rotation=30)
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)

# --- Graphique 2 : Decomposition gamma vs theta cumules ---
ax = axes[0, 1]
ax.set_facecolor(bg_color)
gamma_cumule = df_sim['pnl_gamma_jour'].cumsum()
theta_cumule = df_sim['pnl_theta_jour'].cumsum()
ax.plot(df_sim['Date'], gamma_cumule, color=green,  linewidth=2, label='Gain gamma cumule')
ax.plot(df_sim['Date'], theta_cumule, color=pink,   linewidth=2, label='Cout theta cumule')
ax.plot(df_sim['Date'], gamma_cumule + theta_cumule, color=yellow, linewidth=1.5, linestyle='--', label='Net')
ax.axhline(y=0, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax.set_title('Decomposition Gamma vs Theta', color=cyan, fontsize=12, fontweight='bold')
ax.set_ylabel('P&L cumule', color='white')
ax.tick_params(colors='white', labelsize=8)
ax.xaxis.set_tick_params(rotation=30)
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)
ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=8)

# --- Graphique 3 : Vol implicite vs mouvement journalier absolu ---
ax = axes[1, 0]
ax.set_facecolor(bg_color)
ax2 = ax.twinx()
ax.plot(df_sim['Date'], df_sim['sigma_impl']*100, color=cyan, linewidth=1.5, label='Vol implicite (VIX)')
ax2.bar(df_sim['Date'], df_sim['pnl_jour'], color=[green if p>0 else pink for p in df_sim['pnl_jour']], alpha=0.4, width=1.0)
ax.set_title('Vol implicite et P&L journalier', color=cyan, fontsize=12, fontweight='bold')
ax.set_ylabel('VIX (%)', color=cyan)
ax2.set_ylabel('P&L jour', color='white')
ax.tick_params(colors='white', labelsize=8)
ax2.tick_params(colors='white', labelsize=8)
ax.xaxis.set_tick_params(rotation=30)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)

# --- Graphique 4 : P&L jour vs |dS| (mouvement du marche) ---
ax = axes[1, 1]
ax.set_facecolor(bg_color)
ax.scatter(abs(df_sim['dS']), df_sim['pnl_jour'], color=cyan, alpha=0.4, s=15)
ax.axhline(y=0, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax.set_title('P&L jour vs Mouvement du marche', color=cyan, fontsize=12, fontweight='bold')
ax.set_xlabel('|Variation prix SP500|', color='white')
ax.set_ylabel('P&L jour', color='white')
ax.tick_params(colors='white', labelsize=8)
ax.yaxis.grid(True, color=grid_color, linewidth=0.5)
for spine in ax.spines.values():
    spine.set_edgecolor(grid_color)

fig.suptitle(
    f'Long Straddle Delta-Hedge   |   P&L total: {pnl_total:.1f}   Gamma: {pnl_gamma_total:.1f}   Theta: {pnl_theta_total:.1f}',
    color=yellow, fontsize=12, fontweight='bold'
)

plt.tight_layout()
plt.savefig('sp500_long_straddle_hedge.png', dpi=150, bbox_inches='tight', facecolor=bg_color)
plt.show()
print("Graphique sauvegarde : sp500_long_straddle_hedge.png")

# ============================================================
# ISOLER LES JOURS GAGNANTS DE LA STRATEGIE LONG STRADDLE
# ============================================================

jours_gagnants = df_sim[df_sim['pnl_jour'] > 0].copy()
jours_perdants = df_sim[df_sim['pnl_jour'] <= 0].copy()

print("=" * 60)
print("ANALYSE DES JOURS GAGNANTS — LONG STRADDLE")
print("=" * 60)
print(f"Jours gagnants : {len(jours_gagnants)} ({len(jours_gagnants)/len(df_sim):.1%})")
print(f"Jours perdants : {len(jours_perdants)} ({len(jours_perdants)/len(df_sim):.1%})")
print()

# Statistiques comparatives
print(f"{'':30} {'Gagnants':>12} {'Perdants':>12}")
print("-" * 56)
print(f"{'Mouvement moyen |dS|':30} {abs(jours_gagnants['dS']).mean():>12.2f} {abs(jours_perdants['dS']).mean():>12.2f}")
print(f"{'Mouvement median |dS|':30} {abs(jours_gagnants['dS']).median():>12.2f} {abs(jours_perdants['dS']).median():>12.2f}")
print(f"{'Vol implicite moyenne':30} {jours_gagnants['sigma_impl'].mean():>11.2%} {jours_perdants['sigma_impl'].mean():>11.2%}")
print(f"{'PnL moyen':30} {jours_gagnants['pnl_jour'].mean():>12.2f} {jours_perdants['pnl_jour'].mean():>12.2f}")
print()

# Les 15 meilleurs jours
top15 = df_sim.nlargest(15, 'pnl_jour')[['Date', 'S', 'dS', 'sigma_impl', 'pnl_jour', 'pnl_gamma_jour', 'pnl_theta_jour']]
print("TOP 15 MEILLEURS JOURS :")
print(top15.to_string(index=False))
print()

# Les 15 pires jours
bottom15 = df_sim.nsmallest(15, 'pnl_jour')[['Date', 'S', 'dS', 'sigma_impl', 'pnl_jour', 'pnl_gamma_jour', 'pnl_theta_jour']]
print("TOP 15 PIRES JOURS :")
print(bottom15.to_string(index=False))

# ============================================================
# GROUPER LES JOURS GAGNANTS PAR PERIODE (clusters)
# Pour voir si ce sont des evenements de stress regroupes
# ============================================================

jours_gagnants_dates = jours_gagnants['Date'].sort_values().reset_index(drop=True)

print()
print("=" * 60)
print("CLUSTERS DE JOURS GAGNANTS (ecart > 5 jours = nouveau cluster)")
print("=" * 60)

clusters = []
cluster_actuel = [jours_gagnants_dates.iloc[0]]

for i in range(1, len(jours_gagnants_dates)):
    ecart = (jours_gagnants_dates.iloc[i] - jours_gagnants_dates.iloc[i-1]).days
    if ecart <= 5:
        cluster_actuel.append(jours_gagnants_dates.iloc[i])
    else:
        clusters.append(cluster_actuel)
        cluster_actuel = [jours_gagnants_dates.iloc[i]]
clusters.append(cluster_actuel)

# Trier les clusters par taille et afficher les plus gros
clusters_tries = sorted(clusters, key=len, reverse=True)

print(f"Nombre de clusters identifies : {len(clusters)}")
print()
print("Les 10 plus gros clusters de jours gagnants :")
for c in clusters_tries[:10]:
    debut = c[0].date()
    fin   = c[-1].date()
    print(f"  {debut} -> {fin}  ({len(c)} jours)")

# ============================================================
# GRAPHIQUE — VISUALISER LES CLUSTERS SUR LE PRIX SP500
# ============================================================

plt.style.use('dark_background')
bg_color   = "#0B0E14"
cyan       = "#00F0FF"
green      = "#00FF66"
pink       = "#FF007F"
yellow     = "#FFD700"
grid_color = "#1E2A38"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
fig.patch.set_facecolor(bg_color)

# --- Graphique 1 : Prix SP500 avec jours gagnants surligns ---
ax1.set_facecolor(bg_color)
ax1.plot(df['Date'], df['Price'], color=cyan, linewidth=1.2, alpha=0.7, label='SP500')

for date_g in jours_gagnants['Date']:
    ax1.axvline(x=date_g, color=green, linewidth=1.5, alpha=0.15)

ax1.set_title('SP500 — Jours ou Long Straddle est gagnant (vert)', color=cyan, fontsize=13, fontweight='bold')
ax1.set_ylabel('Prix', color='white')
ax1.tick_params(colors='white', labelsize=9)
ax1.yaxis.grid(True, color=grid_color, linewidth=0.5)
for spine in ax1.spines.values():
    spine.set_edgecolor(grid_color)

# --- Graphique 2 : VIX avec jours gagnants surligns ---
ax2.set_facecolor(bg_color)
ax2.plot(df['Date'], df['VIX'], color=orange, linewidth=1.2, label='VIX')
ax2.axhline(y=df['VIX'].mean(), color='white', linewidth=0.8, linestyle='--', alpha=0.4, label='VIX moyen')

for date_g in jours_gagnants['Date']:
    ax2.axvline(x=date_g, color=green, linewidth=1.5, alpha=0.15)

ax2.set_title('VIX — Memes jours gagnants surlignes', color=cyan, fontsize=13, fontweight='bold')
ax2.set_ylabel('VIX', color='white')
ax2.set_xlabel('Date', color='white')
ax2.tick_params(colors='white', labelsize=9)
ax2.xaxis.set_tick_params(rotation=30)
ax2.yaxis.grid(True, color=grid_color, linewidth=0.5)
for spine in ax2.spines.values():
    spine.set_edgecolor(grid_color)
ax2.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor='white', fontsize=9)

plt.tight_layout()
plt.savefig('sp500_clusters_gagnants.png', dpi=150, bbox_inches='tight', facecolor=bg_color)
plt.show()
print()
print("Graphique sauvegarde : sp500_clusters_gagnants.png")