import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_tabular_benchmarks():
    datasets = ['yacht', 'energy', 'concrete']
    objectives = ['superquantile', 'extremile', 'esrm']
    obj_names = ['CVaR', 'Extremile', 'ESRM']
    
    optimizers = ['sgd', 'srda', 'lsvrg', 'prospect']
    colors = {'sgd': 'black', 'srda': 'gray', 'lsvrg': 'teal', 'prospect': 'red'}
    markers = {'sgd': 'v', 'srda': '.', 'lsvrg': 'o', 'prospect': '^'}

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i, obj in enumerate(objectives):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            base_dir = f'results/{dataset}'
            
            # Find the correct Wasserstein model directory for this objective
            model_dir = None
            if os.path.exists(base_dir):
                for d in os.listdir(base_dir):
                    if 'wasserstein' in d and 'euclidean' in d and f'objective_{obj}' in d:
                        model_dir = os.path.join(base_dir, d)
                        break
                    
            if model_dir is None or not os.path.isdir(model_dir):
                print(f'Missing results for {dataset} - {obj}')
                continue
                
            lbfgs_file = os.path.join(model_dir, 'lbfgs_min_loss.p')
            if not os.path.exists(lbfgs_file):
                print(f'Missing lbfgs file for {dataset} - {obj}')
                continue
                
            with open(lbfgs_file, 'rb') as f:
                try:
                    min_loss = pickle.load(f)
                except EOFError:
                    print(f'Corrupted/empty lbfgs file for {dataset} - {obj}')
                    continue
                
            for opt in optimizers:
                opt_path = os.path.join(model_dir, opt)
                traj_file = os.path.join(opt_path, 'best_traj.p')
                if not os.path.exists(traj_file):
                    print(f'Missing {opt} for {dataset} - {obj}')
                    continue
                    
                with open(traj_file, 'rb') as f:
                    df = pickle.load(f)
                    
                epochs = df['epoch']
                
                # Handling epoch multiplication for LSVRG (LSVRG does 2 passes per epoch)
                x_axis = np.array(epochs) * 2 if opt == 'lsvrg' else np.array(epochs)
                
                train_loss = np.array(df['average_train_loss'])
                init_loss = train_loss[0]
                
                # Suboptimality formula
                eps = 1e-9
                subopt = (train_loss - min_loss + eps) / (init_loss - min_loss + eps)
                
                # Truncate to 125 passes
                mask = x_axis <= 125
                x_axis = x_axis[mask]
                subopt = subopt[mask]
                
                if opt == 'prospect':
                    label_str = 'Prospect (Ours)'
                elif opt == 'lsvrg':
                    label_str = 'LSVRG'
                else:
                    label_str = opt.upper()
                    
                ax.plot(x_axis, subopt, color=colors[opt], marker=markers[opt], 
                        label=label_str, 
                        linewidth=2, markersize=6, markevery=15)
                
            ax.set_yscale('log')
            if i == 0:
                ax.set_title(dataset, fontsize=18)
            if j == 0:
                ax.set_ylabel(obj_names[i], fontsize=16)
            
            if i == 2:
                ax.set_xlabel('Passes', fontsize=14)

    # Create a single legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=16)

    save_path = 'figures/tabular_wasserstein_benchmarks.png'
    os.makedirs('figures', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'Plot successfully saved to {save_path}')

if __name__ == "__main__":
    plot_tabular_benchmarks()
