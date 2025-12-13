#!/usr/bin/env python3
"""
Generate plots and diagrams for parallel matrix multiplication documentation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
import seaborn as sns

# Set style for academic publications
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6

# ============================================================================
# 1. OpenMP Naive Performance Comparison
# ============================================================================
def plot_openmp_naive_performance():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Data for each machine
    machines = {
        'Machine 1': {
            '100x100': [0.0001, 0.0001, 0.0002, 0.0003, 0.0004],
            '1000x1000': [0.0797, 0.0355, 0.0204, 0.0204, 0.0107],
            '10000x10000': [64.6434, 32.6183, 16.1950, 9.7749, 8.3111]
        },
        'Machine 2': {
            '100x100': [0.0001, 0.0002, 0.0004, 0.0140, 0.0014],
            '1000x1000': [0.1230, 0.0712, 0.0474, 0.0698, 0.0377],
            '10000x10000': [149.0415, 93.5300, 76.8988, 75.5792, 81.1862]
        },
        'Machine 3': {
            '100x100': [0.0001, 0.0003, 0.0009, 0.0004, 0.0017],
            '1000x1000': [0.0761, 0.0576, 0.0338, 0.0273, 0.0237],
            '10000x10000': [97.1157, 70.1387, 73.1689, 82.2075, 81.3105]
        }
    }
    
    threads = [1, 2, 4, 8, 16]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (machine_name, data) in enumerate(machines.items()):
        ax = axes[idx]
        
        for size, times in data.items():
            ax.plot(threads, times, marker='o', linewidth=2, 
                   label=size, markersize=6)
        
        ax.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_title(f'{machine_name}\nOpenMP Naive Performance', 
                    fontsize=12, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xticks(threads)
        ax.set_xticklabels(threads)
    
    plt.tight_layout()
    plt.savefig('openmp_naive_performance.png', bbox_inches='tight')
    print("✓ Generated: openmp_naive_performance.png")
    plt.close()

# ============================================================================
# 2. Speedup Analysis
# ============================================================================
def plot_speedup_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Machine 1 - Best performance
    threads = np.array([1, 2, 4, 8, 16])
    
    # 1000x1000 matrix
    times_1000 = {
        'Machine 1': np.array([0.0797, 0.0355, 0.0204, 0.0204, 0.0107]),
        'Machine 2': np.array([0.1230, 0.0712, 0.0474, 0.0698, 0.0377]),
        'Machine 3': np.array([0.0761, 0.0576, 0.0338, 0.0273, 0.0237])
    }
    
    # 10000x10000 matrix
    times_10000 = {
        'Machine 1': np.array([64.6434, 32.6183, 16.1950, 9.7749, 8.3111]),
        'Machine 2': np.array([149.0415, 93.5300, 76.8988, 75.5792, 81.1862]),
        'Machine 3': np.array([97.1157, 70.1387, 73.1689, 82.2075, 81.3105])
    }
    
    # Plot 1000x1000
    ax = axes[0]
    for machine, times in times_1000.items():
        speedup = times[0] / times
        ax.plot(threads, speedup, marker='o', linewidth=2, 
               label=machine, markersize=7)
    
    # Ideal speedup line
    ax.plot(threads, threads, 'k--', linewidth=2, label='Ideal Linear', alpha=0.5)
    ax.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=11, fontweight='bold')
    ax.set_title('Speedup Analysis - 1000×1000 Matrix', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks(threads)
    
    # Plot 10000x10000
    ax = axes[1]
    for machine, times in times_10000.items():
        speedup = times[0] / times
        ax.plot(threads, speedup, marker='o', linewidth=2, 
               label=machine, markersize=7)
    
    ax.plot(threads, threads, 'k--', linewidth=2, label='Ideal Linear', alpha=0.5)
    ax.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=11, fontweight='bold')
    ax.set_title('Speedup Analysis - 10000×10000 Matrix', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks(threads)
    
    plt.tight_layout()
    plt.savefig('speedup_analysis.png', bbox_inches='tight')
    print("✓ Generated: speedup_analysis.png")
    plt.close()

# ============================================================================
# 3. OpenMP Strassen Performance
# ============================================================================
def plot_strassen_performance():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    threads = [1, 2, 4, 8, 16]
    
    # 1000x1000 data
    data_1000 = {
        'Machine 1': [0.0571, 0.0403, 0.0255, 0.0183, 0.0189],
        'Machine 2': [0.1093, 0.0852, 0.0724, 0.0678, 0.0539],
        'Machine 3': [0.0823, 0.0507, 0.0434, 0.0319, 0.0378]
    }
    
    # 10000x10000 data
    data_10000 = {
        'Machine 1': [32.9442, 19.3577, 10.2281, 5.7473, 4.6332],
        'Machine 2': [66.9766, 57.8701, 40.4281, 51.1821, 52.5861],
        'Machine 3': [42.3458, 33.3096, 24.5504, 19.9771, 17.5197]
    }
    
    # Plot 1000x1000
    ax = axes[0]
    for machine, times in data_1000.items():
        ax.plot(threads, times, marker='s', linewidth=2, 
               label=machine, markersize=7)
    
    ax.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Strassen Algorithm - 1000×1000 Matrix', 
                fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks(threads)
    ax.set_xticklabels(threads)
    
    # Plot 10000x10000
    ax = axes[1]
    for machine, times in data_10000.items():
        ax.plot(threads, times, marker='s', linewidth=2, 
               label=machine, markersize=7)
    
    ax.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Strassen Algorithm - 10000×10000 Matrix', 
                fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks(threads)
    ax.set_xticklabels(threads)
    
    plt.tight_layout()
    plt.savefig('strassen_performance.png', bbox_inches='tight')
    print("✓ Generated: strassen_performance.png")
    plt.close()

# ============================================================================
# 4. Algorithm Comparison (Naive vs Strassen)
# ============================================================================
def plot_algorithm_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    threads = [1, 2, 4, 8, 16]
    
    # Machine 1 data for 10000x10000
    naive_times = [64.6434, 32.6183, 16.1950, 9.7749, 8.3111]
    strassen_times = [32.9442, 19.3577, 10.2281, 5.7473, 4.6332]
    
    x = np.arange(len(threads))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, naive_times, width, label='Naive Algorithm',
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, strassen_times, width, label='Strassen Algorithm',
                   color='#F18F01', alpha=0.8)
    
    ax.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Comparison - 10000×10000 Matrix (Machine 1)', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(threads)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', bbox_inches='tight')
    print("✓ Generated: algorithm_comparison.png")
    plt.close()

# ============================================================================
# 5. MPI Naive Algorithm Diagram
# ============================================================================
def draw_mpi_naive_diagram():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.7, 'MPI Naive Matrix Multiplication Architecture', 
            fontsize=16, fontweight='bold', ha='center', family='serif')
    
    # Master process (Rank 0)
    master = FancyBboxPatch((0.5, 7), 2, 1.5, boxstyle="round,pad=0.1",
                           edgecolor='#1f4788', facecolor='#d0e1f9', linewidth=2.5)
    ax.add_patch(master)
    ax.text(1.5, 7.75, 'Rank 0\n(Master Process)', ha='center', va='center',
           fontsize=11, fontweight='bold', family='serif')
    
    # Matrix A and B
    matrix_a = Rectangle((3.5, 7.5), 0.8, 1, facecolor='#8da0cb', alpha=0.6, 
                        edgecolor='black', linewidth=1.5)
    matrix_b = Rectangle((4.5, 7.5), 0.8, 1, facecolor='#fc8d62', alpha=0.6,
                        edgecolor='black', linewidth=1.5)
    ax.add_patch(matrix_a)
    ax.add_patch(matrix_b)
    ax.text(3.9, 8.8, 'Matrix A', ha='center', fontsize=11, fontweight='bold', family='serif')
    ax.text(4.9, 8.8, 'Matrix B', ha='center', fontsize=11, fontweight='bold', family='serif')
    
    # Scatter and Broadcast
    ax.annotate('', xy=(1.5, 5.5), xytext=(1.5, 7),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#8da0cb'))
    ax.text(2.5, 6.2, 'MPI_Scatter\n(distribute rows of A)', fontsize=10, 
           color='#1f4788', style='italic', family='serif',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#8da0cb', linewidth=1.5))
    
    ax.annotate('', xy=(7, 5.5), xytext=(4.9, 7.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#fc8d62'))
    ax.text(6.8, 6.5, 'MPI_Bcast\n(broadcast matrix B)', fontsize=10, 
           color='#1f4788', style='italic', family='serif',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#fc8d62', linewidth=1.5))
    
    # Worker processes
    worker_positions = [(0.3, 4), (2.5, 4), (4.7, 4), (6.9, 4)]
    for i, (x, y) in enumerate(worker_positions):
        worker = FancyBboxPatch((x, y), 1.5, 1, boxstyle="round,pad=0.05",
                               edgecolor='#1f4788', facecolor='#e6f2ff', linewidth=2)
        ax.add_patch(worker)
        ax.text(x + 0.75, y + 0.7, f'Rank {i}', ha='center', fontweight='bold', 
               fontsize=10, family='serif')
        ax.text(x + 0.75, y + 0.3, f'Process {i}\nComputes rows {i*2}–{i*2+1}', 
               ha='center', fontsize=8, family='serif')
    
    # Local computation
    ax.text(5, 3, 'Local Matrix Multiplication (ijk-loop ordering)', 
           ha='center', fontsize=10, style='italic', family='serif',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff9e6', 
                    edgecolor='#666', linewidth=1.5, alpha=0.9))
    
    # Gather
    for i, (x, y) in enumerate(worker_positions):
        ax.annotate('', xy=(4, 1.5), xytext=(x + 0.75, y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#1f4788'))
    
    ax.text(5.8, 1.9, 'MPI_Gather\n(collect results)', fontsize=10, color='#1f4788',
           style='italic', family='serif',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor='#1f4788', linewidth=1.5))
    
    # Result matrix
    result = Rectangle((3.5, 0.3), 1.5, 0.9, facecolor='#a6d96a', alpha=0.7,
                      edgecolor='black', linewidth=1.5)
    ax.add_patch(result)
    ax.text(4.25, 0.75, 'Result Matrix C', ha='center', fontsize=11, 
           fontweight='bold', family='serif')
    
    plt.tight_layout()
    plt.savefig('mpi_naive_diagram.png', bbox_inches='tight')
    print("✓ Generated: mpi_naive_diagram.png")
    plt.close()

# ============================================================================
# 6. MPI Strassen Algorithm Diagram
# ============================================================================
def draw_mpi_strassen_diagram():
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.7, 'MPI Strassen Algorithm: Distributed Computation', 
            fontsize=16, fontweight='bold', ha='center', family='serif')
    
    # Matrix partitioning
    ax.text(1.5, 8.5, 'Input Matrix Partitioning:', fontsize=12, fontweight='bold', family='serif')
    
    # Matrix A partitions - using academic color scheme
    colors_a = ['#b3cde3', '#ccebc5']  # Softer, professional colors
    labels_a = [['A₁₁', 'A₁₂'], ['A₂₁', 'A₂₂']]
    ax.text(0.75, 8.1, 'Matrix A', fontsize=10, fontweight='bold', family='serif')
    for i in range(2):
        for j in range(2):
            rect = Rectangle((0.5 + j*0.5, 7.5 - i*0.5), 0.45, 0.45,
                           facecolor=colors_a[i], edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(0.5 + j*0.5 + 0.225, 7.5 - i*0.5 + 0.225, 
                   labels_a[i][j], ha='center', va='center', fontsize=11, 
                   fontweight='bold', family='serif')
    
    # Matrix B partitions
    labels_b = [['B₁₁', 'B₁₂'], ['B₂₁', 'B₂₂']]
    ax.text(2.25, 8.1, 'Matrix B', fontsize=10, fontweight='bold', family='serif')
    for i in range(2):
        for j in range(2):
            rect = Rectangle((2 + j*0.5, 7.5 - i*0.5), 0.45, 0.45,
                           facecolor=colors_a[i], edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(2 + j*0.5 + 0.225, 7.5 - i*0.5 + 0.225, 
                   labels_b[i][j], ha='center', va='center', fontsize=11,
                   fontweight='bold', family='serif')
    
    # 7 Processes computing M1-M7
    products = [
        'M₁ = (A₁₁+A₂₂)(B₁₁+B₂₂)',
        'M₂ = (A₂₁+A₂₂)B₁₁',
        'M₃ = A₁₁(B₁₂-B₂₂)',
        'M₄ = A₂₂(B₂₁-B₁₁)',
        'M₅ = (A₁₁+A₁₂)B₂₂',
        'M₆ = (A₂₁-A₁₁)(B₁₁+B₁₂)',
        'M₇ = (A₁₂-A₂₂)(B₂₁+B₂₂)'
    ]
    
    # Professional color palette
    colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69']
    
    # Draw processes in two rows
    y_positions = [5.5, 3.5]
    x_start = 0.5
    
    ax.text(6, 6.3, 'Parallel Computation (7 MPI Processes)', fontsize=11, 
           fontweight='bold', ha='center', family='serif')
    
    for i, product in enumerate(products):
        row = 0 if i < 4 else 1
        col = i if i < 4 else i - 4
        x = x_start + col * 2.8
        y = y_positions[row]
        
        # Process box
        box = FancyBboxPatch((x, y), 2.5, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=colors[i], 
                            linewidth=2, alpha=0.8)
        ax.add_patch(box)
        
        # Text
        rank_num = i + 1 if i < 6 else 0
        ax.text(x + 1.25, y + 0.55, f'Process {rank_num}', 
               ha='center', fontweight='bold', fontsize=9, family='serif')
        ax.text(x + 1.25, y + 0.25, product, ha='center', fontsize=9, family='serif')
    
    # MPI Communication annotations
    ax.annotate('', xy=(1, 5.3), xytext=(1, 7.4),
               arrowprops=dict(arrowstyle='->', lw=2, color='#1f4788'))
    ax.text(0.3, 6.3, 'Distribute', ha='center', fontsize=9, style='italic',
           family='serif', rotation=90)
    
    ax.annotate('', xy=(4, 0.8), xytext=(4, 3.3),
               arrowprops=dict(arrowstyle='->', lw=2, color='#1f4788'))
    ax.text(4.7, 2.1, 'Collect', ha='center', fontsize=9, style='italic',
           family='serif', rotation=90)
    
    # Result combination
    result_box = FancyBboxPatch((2.5, 0.5), 7, 1.4, boxstyle="round,pad=0.1",
                               edgecolor='#1f4788', facecolor='#d0e1f9', 
                               linewidth=2.5)
    ax.add_patch(result_box)
    
    ax.text(6, 1.6, 'Final Combination (Process 0):', ha='center', 
           fontweight='bold', fontsize=11, family='serif')
    ax.text(6, 1.2, 'C₁₁ = M₁ + M₄ − M₅ + M₇     C₁₂ = M₃ + M₅', 
           ha='center', fontsize=9, family='serif')
    ax.text(6, 0.85, 'C₂₁ = M₂ + M₄     C₂₂ = M₁ − M₂ + M₃ + M₆', 
           ha='center', fontsize=9, family='serif')
    
    plt.tight_layout()
    plt.savefig('mpi_strassen_diagram.png', bbox_inches='tight')
    print("✓ Generated: mpi_strassen_diagram.png")
    plt.close()

# ============================================================================
# 7. OpenMP Task Parallelism Diagram
# ============================================================================
def draw_openmp_task_diagram():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.7, 'OpenMP Task-Based Parallelism Architecture', 
            fontsize=16, fontweight='bold', ha='center', family='serif')
    
    # Main thread
    main_thread = FancyBboxPatch((4, 8), 4, 0.9, boxstyle="round,pad=0.1",
                                edgecolor='#1f4788', facecolor='#d0e1f9', 
                                linewidth=2.5)
    ax.add_patch(main_thread)
    ax.text(6, 8.5, 'Master Thread', ha='center', va='center', 
           fontsize=12, fontweight='bold', family='serif')
    ax.text(6, 8.15, '#pragma omp single', ha='center', va='center',
           fontsize=9, style='italic', family='monospace')
    
    # Task creation
    ax.text(6, 7.3, 'Task Creation:', ha='center', 
           fontsize=11, fontweight='bold', family='serif')
    ax.text(6, 7.0, '#pragma omp taskgroup', ha='center', 
           fontsize=9, style='italic', family='monospace',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9e6', 
                    edgecolor='#666', linewidth=1.5, alpha=0.9))
    
    # Tasks for 8 multiplications (Naive) or 7 (Strassen)
    task_labels_naive = [
        'A₁₁ × B₁₁ → C₁₁',
        'A₁₁ × B₁₂ → C₁₂',
        'A₂₁ × B₁₁ → C₂₁',
        'A₂₁ × B₁₂ → C₂₂'
    ]
    
    task_labels_naive2 = [
        'A₁₂ × B₂₁ → C₁₁',
        'A₁₂ × B₂₂ → C₁₂',
        'A₂₂ × B₂₁ → C₂₁',
        'A₂₂ × B₂₂ → C₂₂'
    ]
    
    # Professional color palette for tasks
    colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5']
    
    # First wave of tasks
    ax.text(0.8, 6.5, 'First Wave:', fontsize=11, fontweight='bold', family='serif')
    for i, label in enumerate(task_labels_naive):
        x = 1.5 + i * 2.3
        task = FancyBboxPatch((x, 5.5), 2, 0.7, boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=colors[i], 
                             linewidth=2, alpha=0.85)
        ax.add_patch(task)
        ax.text(x + 1, 5.9, f'Task {i+1}', ha='center', 
               fontweight='bold', fontsize=9, family='serif')
        ax.text(x + 1, 5.65, label, ha='center', fontsize=8, family='serif')
    
    # Task wait synchronization
    ax.text(6, 4.9, 'Synchronization Barrier', ha='center', 
           fontsize=10, fontweight='bold', family='serif')
    ax.text(6, 4.6, '#pragma omp taskwait', ha='center', 
           fontsize=9, style='italic', family='monospace',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffd4a3', 
                    edgecolor='#666', linewidth=1.5, alpha=0.9))
    
    # Second wave of tasks
    ax.text(0.8, 4.2, 'Second Wave:', fontsize=11, fontweight='bold', family='serif')
    for i, label in enumerate(task_labels_naive2):
        x = 1.5 + i * 2.3
        task = FancyBboxPatch((x, 3.3), 2, 0.7, boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=colors[i+4], 
                             linewidth=2, alpha=0.85)
        ax.add_patch(task)
        ax.text(x + 1, 3.7, f'Task {i+5}', ha='center', 
               fontweight='bold', fontsize=9, family='serif')
        ax.text(x + 1, 3.45, label, ha='center', fontsize=8, family='serif')
    
    # Thread pool
    ax.text(6, 2.3, 'OpenMP Thread Pool', ha='center', 
           fontsize=11, fontweight='bold', family='serif')
    ax.text(6, 2.05, '(Dynamic Task Scheduling)', ha='center',
           fontsize=9, style='italic', family='serif')
    
    # Professional thread colors
    thread_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
                    '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    for i in range(8):
        x = 1.5 + i * 1.2
        thread = Rectangle((x, 0.9), 0.9, 0.9, facecolor=thread_colors[i], 
                          edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(thread)
        ax.text(x + 0.45, 1.35, f'Thread {i}', ha='center', va='center', 
               fontsize=8, fontweight='bold', family='serif')
    
    # Recursive nature note
    ax.text(6, 0.3, 'Note: Tasks recursively subdivide until base case threshold', 
           ha='center', fontsize=9, style='italic', family='serif',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor='gray', linewidth=1, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('openmp_task_diagram.png', bbox_inches='tight')
    print("✓ Generated: openmp_task_diagram.png")
    plt.close()

# ============================================================================
# 8. Hybrid MPI+OpenMP Architecture
# ============================================================================
def draw_hybrid_architecture():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.7, 'Hybrid MPI+OpenMP Parallel Architecture', 
            fontsize=16, fontweight='bold', ha='center', family='serif')
    
    # Network layer
    network = Rectangle((0.2, 6.8), 11.6, 0.35, facecolor='#ffd92f', 
                       edgecolor='black', linewidth=2.5, alpha=0.7)
    ax.add_patch(network)
    ax.text(6, 6.98, 'MPI Communication Layer', 
           ha='center', va='center', fontsize=11, fontweight='bold', family='serif')
    ax.text(6, 6.78, '(Distributed Memory / Inter-Process)', 
           ha='center', va='center', fontsize=8, style='italic', family='serif')
    
    # 7 MPI Processes
    process_width = 1.5
    process_height = 2.1
    spacing = 0.1
    start_x = 0.5
    
    for i in range(7):
        x = start_x + i * (process_width + spacing)
        
        # Process box
        process = FancyBboxPatch((x, 3.9), process_width, process_height,
                                boxstyle="round,pad=0.05",
                                edgecolor='#1f4788', facecolor='#d0e1f9',
                                linewidth=2.5)
        ax.add_patch(process)
        
        # Process label
        rank = i + 1 if i < 6 else 0
        ax.text(x + process_width/2, 5.7, f'Process {rank}',
               ha='center', fontweight='bold', fontsize=9, family='serif')
        
        # OpenMP threads within each process
        thread_colors = ['#e78ac3', '#8da0cb', '#a6d854', '#ffd92f']
        for j in range(4):
            thread_y = 4.15 + j * 0.37
            thread = Rectangle((x + 0.15, thread_y), process_width - 0.3, 0.32,
                             facecolor=thread_colors[j], edgecolor='black',
                             linewidth=1.5, alpha=0.85)
            ax.add_patch(thread)
            ax.text(x + process_width/2, thread_y + 0.16, f'T{j}',
                   ha='center', fontsize=7, fontweight='bold', family='serif')
    
    # OpenMP layer label
    ax.text(6, 3.4, 'OpenMP Layer', ha='center', fontsize=11, fontweight='bold',
           family='serif',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#a6d96a', 
                    edgecolor='black', linewidth=2, alpha=0.7))
    ax.text(6, 3.1, '(Shared Memory / Intra-Process Threading)', ha='center',
           fontsize=8, style='italic', family='serif')
    
    # Workflow
    workflow_y = 2.4
    ax.text(6, workflow_y, 'Hybrid Execution Model:', ha='center', 
           fontsize=12, fontweight='bold', family='serif')
    
    steps = [
        '① MPI distributes Strassen submatrices (M₁–M₇) across 7 processes',
        '② Each MPI process spawns OpenMP threads for local computation',
        '③ OpenMP threads perform parallel matrix operations using shared memory',
        '④ Processes synchronize and gather partial results via MPI',
        '⑤ Master process (Rank 0) combines results into final matrix'
    ]
    
    for i, step in enumerate(steps):
        ax.text(6, workflow_y - 0.35 - i*0.28, step, ha='center', 
               fontsize=8.5, family='serif')
    
    # Performance note
    ax.text(6, 0.25, 'Two-level parallelism: Coarse-grained (MPI) + Fine-grained (OpenMP)',
           ha='center', fontsize=9, family='serif',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff9e6', 
                    edgecolor='#666', linewidth=1.5, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('hybrid_architecture.png', bbox_inches='tight')
    print("✓ Generated: hybrid_architecture.png")
    plt.close()

# ============================================================================
# 9. Efficiency Comparison Heatmap
# ============================================================================
def plot_efficiency_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    machines = ['Machine 1', 'Machine 2', 'Machine 3']
    threads = [1, 2, 4, 8, 16]
    
    # Efficiency data (speedup / num_threads * 100)
    # Machine 1 - 10000x10000
    speedup_m1 = np.array([1, 1.98, 3.99, 6.61, 7.78])
    efficiency_m1 = (speedup_m1 / np.array(threads)) * 100
    
    # Machine 2 - 10000x10000
    speedup_m2 = np.array([1, 1.59, 1.94, 1.97, 1.84])
    efficiency_m2 = (speedup_m2 / np.array(threads)) * 100
    
    # Machine 3 - 10000x10000
    speedup_m3 = np.array([1, 1.38, 1.33, 1.18, 1.19])
    efficiency_m3 = (speedup_m3 / np.array(threads)) * 100
    
    efficiencies = [efficiency_m1, efficiency_m2, efficiency_m3]
    
    for idx, (ax, machine, eff) in enumerate(zip(axes, machines, efficiencies)):
        # Create heatmap
        data = eff.reshape(5, 1)
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(threads)
        ax.set_xticks([0])
        ax.set_xticklabels(['Efficiency %'])
        
        # Add values
        for i in range(5):
            text = ax.text(0, i, f'{eff[i]:.1f}%',
                          ha="center", va="center", color="black", 
                          fontweight='bold', fontsize=11)
        
        ax.set_title(f'{machine}\n10000×10000 Matrix', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Number of Threads', fontsize=11, fontweight='bold')
    
    # Add colorbar with better positioning - on the right side
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Parallel Efficiency (%)', fontsize=11, fontweight='bold', rotation=270, labelpad=20)
    
    plt.savefig('efficiency_heatmap.png', bbox_inches='tight', dpi=300)
    print("✓ Generated: efficiency_heatmap.png")
    plt.close()

# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    print("Generating plots and diagrams...\n")
    
    # Performance plots
    plot_openmp_naive_performance()
    plot_speedup_analysis()
    plot_strassen_performance()
    plot_algorithm_comparison()
    plot_efficiency_heatmap()
    
    # Architecture diagrams
    draw_mpi_naive_diagram()
    draw_mpi_strassen_diagram()
    draw_openmp_task_diagram()
    draw_hybrid_architecture()
    
    print("\n✓ All plots and diagrams generated successfully!")
    print("\nGenerated files:")
    print("  - openmp_naive_performance.png")
    print("  - speedup_analysis.png")
    print("  - strassen_performance.png")
    print("  - algorithm_comparison.png")
    print("  - efficiency_heatmap.png")
    print("  - mpi_naive_diagram.png")
    print("  - mpi_strassen_diagram.png")
    print("  - openmp_task_diagram.png")
    print("  - hybrid_architecture.png")
