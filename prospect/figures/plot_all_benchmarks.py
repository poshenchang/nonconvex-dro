import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def get_model_dir(dataset, objective):
    base_dir = f'results/{dataset}'
    if not os.path.exists(base_dir): return None
    
    # Try to find wasserstein penalty first, otherwise fallback to any
    found_dir = None
    for d in os.listdir(base_dir):
        # We need to match the objective (e.g. esrm, extremile, superquantile for CVaR)
        target_obj = "superquantile" if objective == "cvar" else objective
        if f'objective_{target_obj}' in d:
            if 'wasserstein' in d and 'euclidean' in d:
                return os.path.join(base_dir, d)
            found_dir = os.path.join(base_dir, d)
    return found_dir

def plot_regression_3x3():
    datasets = ['yacht', 'energy', 'concrete']
    objectives = ['cvar', 'extremile', 'esrm']
    optimizers = ['sgd', 'srda', 'lsvrg', 'prospect']
    colors = {'sgd': 'black', 'srda': 'gray', 'lsvrg': 'teal', 'prospect': 'red'}
    markers = {'sgd': 'v', 'srda': '.', 'lsvrg': 'o', 'prospect': '^'}

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for row, objective in enumerate(objectives):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            model_dir = get_model_dir(dataset, objective)
            
            if model_dir is None or not os.path.isdir(model_dir):
                ax.text(0.5, 0.5, 'Data Unavailable', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
                
            lbfgs_file = os.path.join(model_dir, 'lbfgs_min_loss.p')
            if not os.path.exists(lbfgs_file):
                ax.text(0.5, 0.5, 'No LBFGS min_loss', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
                
            with open(lbfgs_file, 'rb') as f:
                min_loss = pickle.load(f)
                
            plotted = False
            for opt in optimizers:
                opt_path = os.path.join(model_dir, opt)
                traj_file = os.path.join(opt_path, 'best_traj.p')
                if not os.path.exists(traj_file):
                    continue
                    
                with open(traj_file, 'rb') as f:
                    df = pickle.load(f)
                    
                epochs = df['epoch']
                train_loss = np.array(df['average_train_loss'])
                init_loss = train_loss[0]
                
                eps = 1e-9
                subopt = np.maximum(np.abs(train_loss - min_loss), eps) / np.maximum(np.abs(init_loss - min_loss), eps)
                
                label_str = 'Prospect (Ours)' if opt == 'prospect' else opt.upper()
                ax.plot(epochs, subopt, color=colors[opt], marker=markers[opt], 
                        label=label_str, linewidth=2, markersize=6, markevery=15)
                plotted = True
                
            if not plotted:
                ax.text(0.5, 0.5, 'No Optimizer Data', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ax.set_yscale('log')
            
            if row == 0:
                ax.set_title(dataset, fontsize=16)
            if col == 0:
                ax.set_ylabel(objective.upper(), fontsize=16)
            if row == 2:
                ax.set_xlabel('Passes', fontsize=14)

    handles, labels = [], []
    for ax in axes.flat:
        if ax.get_legend_handles_labels()[0]:
            handles, labels = ax.get_legend_handles_labels()
            break
            
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=14)

    plt.savefig('regression_benchmarks.png', bbox_inches='tight', dpi=300)
    print('Regression benchmarks saved to regression_benchmarks.png')

def plot_diabetes_1x3():
    dataset = 'diabetes'
    objectives = ['cvar', 'extremile', 'esrm']
    optimizers = ['sgd', 'srda', 'lsvrg', 'prospect']
    colors = {'sgd': 'black', 'srda': 'gray', 'lsvrg': 'teal', 'prospect': 'red'}
    markers = {'sgd': 'v', 'srda': '.', 'lsvrg': 'o', 'prospect': '^'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plt.subplots_adjust(wspace=0.3)

    for col, objective in enumerate(objectives):
        ax = axes[col]
        model_dir = get_model_dir(dataset, objective)
        
        # Formatting title as requested
        if objective == 'extremile':
            title_str = 'Extremile'
        elif objective == 'cvar':
            title_str = 'CVaR'
        else:
            title_str = objective.upper()
        
        if model_dir is None or not os.path.isdir(model_dir):
            ax.text(0.5, 0.5, 'Data Unavailable', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title_str, fontsize=16)
            continue
            
        lbfgs_file = os.path.join(model_dir, 'lbfgs_min_loss.p')
        if not os.path.exists(lbfgs_file):
            ax.text(0.5, 0.5, 'No LBFGS min_loss', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title_str, fontsize=16)
            continue
            
        with open(lbfgs_file, 'rb') as f:
            min_loss = pickle.load(f)
            
        plotted = False
        
        for opt in optimizers:
            opt_path = os.path.join(model_dir, opt)
            traj_file = os.path.join(opt_path, 'best_traj.p')
            if not os.path.exists(traj_file):
                continue
                
            with open(traj_file, 'rb') as f:
                df = pickle.load(f)
                
            epochs = df['epoch']
            train_loss = np.array(df['average_train_loss'])
            init_loss = train_loss[0]
            
            # Reverting to 1e-9 and using np.abs to prevent negative log-drops without visual plunging
            eps = 1e-9
            subopt = np.maximum(np.abs(train_loss - min_loss), eps) / np.maximum(np.abs(init_loss - min_loss), eps)
            
            label_str = 'Prospect (Ours)' if opt == 'prospect' else opt.upper()
            ax.plot(epochs, subopt, color=colors[opt], marker=markers[opt], 
                    label=label_str, linewidth=2, markersize=6, markevery=15)
            plotted = True
            
        if not plotted:
            ax.text(0.5, 0.5, 'No Optimizer Data', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title_str, fontsize=16)
            continue

        ax.set_yscale('log')
        ax.set_title(title_str, fontsize=16)
        if col == 0:
            ax.set_ylabel('Suboptimality', fontsize=16)
        ax.set_xlabel('Passes', fontsize=14)

    handles, labels = [], []
    for ax in axes.flat:
        if ax.get_legend_handles_labels()[0]:
            handles, labels = ax.get_legend_handles_labels()
            break
            
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=14)

    plt.savefig('diabetes_benchmarks.png', bbox_inches='tight', dpi=300)
    print('Diabetes benchmarks saved to diabetes_benchmarks.png')

if __name__ == '__main__':
    plot_regression_3x3()
    plot_diabetes_1x3()
