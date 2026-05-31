import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_benchmarks():
    datasets = ['yacht', 'energy', 'concrete']
    optimizers = ['sgd', 'srda', 'prospect']
    colors = {'sgd': 'black', 'srda': 'gray', 'prospect': 'red'}
    markers = {'sgd': 'v', 'srda': '.', 'prospect': '^'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plt.subplots_adjust(wspace=0.3)

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        base_dir = f'results/{dataset}'
        
        # Find the correct Wasserstein model directory
        model_dir = None
        if os.path.exists(base_dir):
            for d in os.listdir(base_dir):
                if 'wasserstein' in d and 'euclidean' in d:
                    model_dir = os.path.join(base_dir, d)
                    break
                
        if model_dir is None or not os.path.isdir(model_dir):
            print(f'Missing results for {dataset}')
            continue
            
        lbfgs_file = os.path.join(model_dir, 'lbfgs_min_loss.p')
        with open(lbfgs_file, 'rb') as f:
            min_loss = pickle.load(f)
            
        for opt in optimizers:
            opt_path = os.path.join(model_dir, opt)
            traj_file = os.path.join(opt_path, 'best_traj.p')
            if not os.path.exists(traj_file):
                print(f'Missing {opt} for {dataset}')
                continue
                
            with open(traj_file, 'rb') as f:
                df = pickle.load(f)
                
            epochs = df['epoch']
            train_loss = np.array(df['average_train_loss'])
            init_loss = train_loss[0]
            
            # Suboptimality formula
            eps = 1e-9
            subopt = (train_loss - min_loss + eps) / (init_loss - min_loss + eps)
            
            ax.plot(epochs, subopt, color=colors[opt], marker=markers[opt], 
                    label=opt.upper() if opt != 'prospect' else 'Prospect (Ours)', 
                    linewidth=2, markersize=6)
            
        ax.set_yscale('log')
        ax.set_title(dataset, fontsize=16)
        if i == 0:
            ax.set_ylabel('Suboptimality (ESRM)', fontsize=14)
        ax.set_xlabel('Passes', fontsize=14)

    # Create a single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=14)

    save_path = 'figures/wasserstein_benchmarks.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'Plot successfully saved to {save_path}')

if __name__ == "__main__":
    plot_benchmarks()
