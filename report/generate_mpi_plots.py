#!/usr/bin/env python3
"""
Generate plots for MPI WireGuard results
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# MPI WireGuard Results Data
mpi_naive_data = {
    '100': {1: 9.2306e-05, 2: 0.0106177, 4: 0.0351109},
    '1000': {1: 0.115776, 2: 0.0834282, 4: 0.134296, 8: 0.252408},
    '4000': {1: 17.6562, 2: 11.372, 4: 9.09094, 8: 10.1447}
}

mpi_strassen_data = {
    '100': {7: 0.148638},
    '4000': {7: 8.01379}
    # Note: 1000x1000 had negative time (error), excluded
}

hybrid_strassen_data = {
    '100': {
        1: 0.133023, 2: 0.163254, 4: 0.609867, 8: 239.778
    },
    '1000': {
        1: 0.179783, 2: 0.435721, 4: 18.8602, 8: 200.497
    },
    '4000': {
        1: 7.33672, 2: 6.2119, 4: 65.4174, 8: 161.198
    }
}

hybrid_naive_data = {
    '100': {
        1: 0.000125892, 2: 8.6394e-05, 4: 6.0607e-05, 8: 0.0185811
    },
    '1000': {
        1: 0.148591, 2: 0.0777425, 4: 0.191566, 8: 0.0504792
    }
}

def plot_mpi_naive_performance():
    """Plot MPI Naive performance across different matrix sizes and process counts"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    matrix_sizes = ['100', '1000', '4000']
    titles = ['100×100', '1000×1000', '4000×4000']
    
    for idx, (size, title) in enumerate(zip(matrix_sizes, titles)):
        ax = axes[idx]
        data = mpi_naive_data[size]
        processes = sorted(data.keys())
        times = [data[p] for p in processes]
        
        ax.plot(processes, times, marker='o', linewidth=2, markersize=8, label='MPI Naive')
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'Matrix Size: {title}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(processes)
        
    plt.tight_layout()
    plt.savefig('mpi_naive_performance.png', bbox_inches='tight')
    print("Generated: mpi_naive_performance.png")
    plt.close()

def plot_mpi_speedup():
    """Plot speedup for MPI Naive implementation"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for size in ['1000', '4000']:
        data = mpi_naive_data[size]
        processes = sorted(data.keys())
        baseline = data[1]
        speedups = [baseline / data[p] for p in processes]
        
        label = f'{size}×{size}'
        ax.plot(processes, speedups, marker='o', linewidth=2, markersize=8, label=label)
    
    # Ideal speedup line
    max_processes = 8
    ax.plot([1, max_processes], [1, max_processes], 'k--', alpha=0.5, label='Ideal Speedup')
    
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Speedup')
    ax.set_title('MPI Naive Speedup Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, max_processes + 0.5)
    ax.set_ylim(0, max_processes + 1)
    
    plt.tight_layout()
    plt.savefig('mpi_speedup_analysis.png', bbox_inches='tight')
    print("Generated: mpi_speedup_analysis.png")
    plt.close()

def plot_hybrid_performance():
    """Plot Hybrid MPI+OpenMP performance"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    matrix_sizes = ['100', '1000', '4000']
    titles = ['100×100', '1000×1000', '4000×4000']
    
    # Strassen plots
    for idx, (size, title) in enumerate(zip(matrix_sizes, titles)):
        ax = axes[0, idx]
        if size in hybrid_strassen_data:
            data = hybrid_strassen_data[size]
            threads = sorted(data.keys())
            times = [data[t] for t in threads]
            
            ax.plot(threads, times, marker='s', linewidth=2, markersize=8, 
                   color='orange', label='Hybrid Strassen')
            ax.set_xlabel('Number of Threads')
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title(f'{title} - Strassen')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(threads)
            ax.legend()
            ax.set_yscale('log')
    
    # Naive plots
    for idx, (size, title) in enumerate(zip(matrix_sizes[:2], titles[:2])):
        ax = axes[1, idx]
        if size in hybrid_naive_data:
            data = hybrid_naive_data[size]
            threads = sorted(data.keys())
            times = [data[t] for t in threads]
            
            ax.plot(threads, times, marker='^', linewidth=2, markersize=8, 
                   color='green', label='Hybrid Naive')
            ax.set_xlabel('Number of Threads')
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title(f'{title} - Naive')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(threads)
            ax.legend()
    
    # Hide unused subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('hybrid_mpi_openmp_performance.png', bbox_inches='tight')
    print("Generated: hybrid_mpi_openmp_performance.png")
    plt.close()

def plot_algorithm_comparison_mpi():
    """Compare MPI Naive vs Strassen vs Hybrid for 4000×4000 matrix"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # MPI Naive data for 4000x4000
    naive_processes = sorted(mpi_naive_data['4000'].keys())
    naive_times = [mpi_naive_data['4000'][p] for p in naive_processes]
    
    # MPI Strassen (7 processes fixed)
    strassen_time = mpi_strassen_data['4000'][7]
    
    # Hybrid Strassen (7 processes, varying threads)
    hybrid_threads = sorted(hybrid_strassen_data['4000'].keys())
    hybrid_times = [hybrid_strassen_data['4000'][t] for t in hybrid_threads]
    
    width = 0.25
    x = np.arange(len(naive_processes))
    
    ax.bar(x - width, naive_times, width, label='MPI Naive', alpha=0.8)
    ax.axhline(y=strassen_time, color='orange', linestyle='--', linewidth=2, 
               label=f'MPI Strassen (7 proc): {strassen_time:.2f}s')
    
    # Plot hybrid times at corresponding positions
    hybrid_x = [list(naive_processes).index(t) if t in naive_processes else -1 
                for t in hybrid_threads]
    valid_hybrid = [(x[i] + width, hybrid_times[i]) 
                    for i in range(len(hybrid_threads)) if hybrid_x[i] >= 0]
    
    if valid_hybrid:
        hx, hy = zip(*valid_hybrid)
        ax.bar(hx, hy, width, label='Hybrid (7 proc + threads)', alpha=0.8, color='green')
    
    ax.set_xlabel('Number of Processes/Threads')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Algorithm Comparison - 4000×4000 Matrix (MPI WireGuard)')
    ax.set_xticks(x)
    ax.set_xticklabels(naive_processes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('mpi_algorithm_comparison.png', bbox_inches='tight')
    print("Generated: mpi_algorithm_comparison.png")
    plt.close()

def plot_efficiency_heatmap_mpi():
    """Plot efficiency heatmap for MPI implementations"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MPI Naive efficiency
    sizes = ['1000', '4000']
    processes_list = [[1, 2, 4, 8], [1, 2, 4, 8]]
    
    efficiency_data = []
    for size, procs in zip(sizes, processes_list):
        row = []
        baseline = mpi_naive_data[size][1]
        for p in procs:
            if p in mpi_naive_data[size]:
                speedup = baseline / mpi_naive_data[size][p]
                efficiency = speedup / p * 100
                row.append(efficiency)
            else:
                row.append(0)
        efficiency_data.append(row)
    
    sns.heatmap(efficiency_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                xticklabels=procs, yticklabels=[f'{s}×{s}' for s in sizes],
                cbar_kws={'label': 'Efficiency (%)'}, ax=axes[0], vmin=0, vmax=100)
    axes[0].set_title('MPI Naive Efficiency')
    axes[0].set_xlabel('Number of Processes')
    axes[0].set_ylabel('Matrix Size')
    
    # Hybrid efficiency
    sizes_hybrid = ['100', '1000', '4000']
    threads = [1, 2, 4, 8]
    
    efficiency_hybrid = []
    for size in sizes_hybrid:
        row = []
        if size in hybrid_strassen_data and 1 in hybrid_strassen_data[size]:
            baseline = hybrid_strassen_data[size][1]
            for t in threads:
                if t in hybrid_strassen_data[size]:
                    speedup = baseline / hybrid_strassen_data[size][t]
                    efficiency = speedup / t * 100
                    row.append(efficiency)
                else:
                    row.append(0)
        else:
            row = [0] * len(threads)
        efficiency_hybrid.append(row)
    
    sns.heatmap(efficiency_hybrid, annot=True, fmt='.1f', cmap='RdYlGn', 
                xticklabels=threads, yticklabels=[f'{s}×{s}' for s in sizes_hybrid],
                cbar_kws={'label': 'Efficiency (%)'}, ax=axes[1], vmin=0, vmax=100)
    axes[1].set_title('Hybrid MPI+OpenMP Efficiency (Strassen)')
    axes[1].set_xlabel('Number of Threads (7 MPI processes)')
    axes[1].set_ylabel('Matrix Size')
    
    plt.tight_layout()
    plt.savefig('mpi_efficiency_heatmap.png', bbox_inches='tight')
    print("Generated: mpi_efficiency_heatmap.png")
    plt.close()

def plot_strong_scaling():
    """Plot strong scaling analysis"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 4000x4000 matrix
    data = mpi_naive_data['4000']
    processes = sorted(data.keys())
    times = [data[p] for p in processes]
    
    # Calculate efficiency
    baseline = data[1]
    efficiencies = [(baseline / (data[p] * p)) * 100 for p in processes]
    
    ax2 = ax.twinx()
    
    # Plot execution time
    color = 'tab:blue'
    ax.plot(processes, times, marker='o', linewidth=2, markersize=8, 
            color=color, label='Execution Time')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Execution Time (seconds)', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True, alpha=0.3)
    
    # Plot efficiency
    color = 'tab:orange'
    ax2.plot(processes, efficiencies, marker='s', linewidth=2, markersize=8, 
             color=color, label='Efficiency', linestyle='--')
    ax2.set_ylabel('Parallel Efficiency (%)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 120)
    
    ax.set_title('Strong Scaling Analysis - 4000×4000 Matrix (MPI Naive)')
    ax.set_xticks(processes)
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('mpi_strong_scaling.png', bbox_inches='tight')
    print("Generated: mpi_strong_scaling.png")
    plt.close()

if __name__ == '__main__':
    print("Generating MPI WireGuard result plots...")
    plot_mpi_naive_performance()
    plot_mpi_speedup()
    plot_hybrid_performance()
    plot_algorithm_comparison_mpi()
    plot_efficiency_heatmap_mpi()
    plot_strong_scaling()
    print("\nAll MPI plots generated successfully!")
