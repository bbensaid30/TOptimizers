import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def read_jsonl(filename):

    df = pd.read_json(filename, lines=True)

    print(len(df))
    print(df.columns)
    return df

#concaténation d'un dataframe pour tous les optis
def preparer_donnees(chemin_dossier):
    fichiers = glob.glob(os.path.join(chemin_dossier, "*.jsonl"))
    df = pd.concat([pd.read_json(f, lines=True) for f in fichiers], ignore_index=True)
    # On filtre les succès et on s'assure que 'algo' est une catégorie pour le tri
    df = df[df['status'] == 'success'].copy()
    return df

def plot_dual_boxplots_article(df, metrique='cost_test'):
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # On définit la largeur des boîtes
    width = 0.35
    algos = df['opti'].unique()
    x_indices = np.arange(len(algos))

    # --- 1. BOXPLOT COST (ROUGE - À GAUCHE) ---
    # On décale la position de -width/2
    box_cost = ax1.boxplot(
        [df[df['opti'] == a][metrique] for a in algos],
        positions=x_indices - width/2,
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        showfliers=True
    )
    
    # Coloration en rouge
    for patch in box_cost['boxes']:
        patch.set_facecolor('red') 

    if metrique in ["cost_train", "cost_test"]:
        ax1.set_yscale('log')
        ax1.set_ylabel('Loss (Log Scale)', fontsize=14, fontweight='bold', color='red')
    elif metrique in ["categorical_accuracy_train","categorical_accuracy_test"]:
        ax1.set_ylabel('% of well classified', fontsize=14, fontweight='bold', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    # --- 2. AXE DROIT ET BOXPLOT TIME (BLEU - À DROITE) ---
    ax2 = ax1.twinx()
    
    # On décale la position de +width/2
    box_time = ax2.boxplot(
        [df[df['opti'] == a]['time_train'] for a in algos],
        positions=x_indices + width/2,
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        showfliers=True
    )
    
    # Coloration en bleu
    for patch in box_time['boxes']:
        patch.set_facecolor('blue')

    ax2.set_ylabel('time (s.)', color="blue", fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')

    # --- CONFIGURATION DES AXES ---
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(algos, fontweight='bold')
    #ax1.set_xlabel('Optimizers', fontsize=14, fontweight='bold')

    #plt.title('Benchmark Results: Loss and Computational Cost', fontsize=16, pad=20)
    ax1.xaxis.grid(False) # Pas de grille verticale
    
    plt.tight_layout()
    plt.show()

def plot_performance_profile(df, loss_col='cost_test'):
    plt.figure(figsize=(12, 8))
    sns.set_style("ticks")
    
    # 1. Définition des familles et de l'ordre de la légende
    # L'ordre dans cette liste déterminera l'ordre dans la légende
    familles_ordre = ['ADAM', 'RMS', 'MOM', 'PGD', 'GD']
    
    color_map = {
        'ADAM': '#d62728',      # Rouge
        'RMS': '#1f77b4',       # Bleu
        'MOMENTUM': '#2ca02c',  # Vert
        'PGD': '#17becf',       # Cyan
        'GD': '#ff7f0e',        # Orange
        'DEFAULT': '#7f7f7f'
    }

    # 2. Fonction de tri à double niveau
    def get_sort_key(algo):
        algo_up = algo.upper()
        
        # Niveau 1 : Index de la famille
        famille_idx = len(familles_ordre)
        for i, famille in enumerate(familles_ordre):
            if famille in algo_up:
                famille_idx = i
                break
        
        # Niveau 2 : Version (Standard < LC < LCD)
        # On donne un poids : 0 pour standard, 1 pour LC, 2 pour LCD
        version_idx = 0
        if 'LCD' in algo_up:
            version_idx = 2
        elif 'LC' in algo_up:
            version_idx = 1
            
        return (famille_idx, version_idx)

    # Tri des algos selon les deux critères
    algos_tries = sorted(df['opti'].unique(), key=get_sort_key)

    alphas = np.linspace(-6, 1, 200)

    # 3. Boucle de tracé
    for algo in algos_tries:
        algo_up = algo.upper()
        
        # --- LOGIQUE COULEUR (Ordre inverse pour éviter les faux positifs) ---
        color = color_map['DEFAULT']
        if 'ADAM' in algo_up: color = color_map['ADAM']
        elif 'RMS' in algo_up: color = color_map['RMS']
        elif 'MOM' in algo_up: color = color_map['MOMENTUM']
        elif 'NGD' in algo_up: color = color_map['PGD']
        elif 'GD' in algo_up: color = color_map['GD']

        # --- LOGIQUE LINESTYLE (Figure 8) ---
        if 'LCD' in algo_up:
            style = '-.' # dash-dot
        elif 'LC' in algo_up:
            style = '--' # dashed
        else:
            style = '-'  # solid

        # Calcul des proportions
        data_algo = df[df['opti'] == algo][loss_col].values
        data_algo = data_algo[~np.isnan(data_algo)]
        proportions = [np.mean(data_algo <= 10**a) for a in alphas]
        
        plt.plot(alphas, proportions, 
                 label=algo, 
                 color=color, 
                 linestyle=style, 
                 linewidth=2.5, 
                 alpha=0.8)

    # Configuration finale
    plt.xlabel(r'$\log_{10}(error)$', fontsize=14, fontweight='bold')
    plt.ylabel(r'probability for the loss of being under $\log_{10}(error)$', fontsize=14)
    
    # Légende groupée
    plt.legend(loc='upper left', frameon=True, fontsize=10, ncol=2)
    
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.ylim(-0.02, 1.02)
    plt.xlim(-6, 1)
    
    sns.despine()
    plt.tight_layout()
    plt.show()

folder="FASHION_MNIST/"
df=preparer_donnees(folder)

#print(df[df['opti'] == "Momentum"]["cost_train"])
plot_dual_boxplots_article(df,"categorical_accuracy_test")
#plot_performance_profile(df,"cost_train")