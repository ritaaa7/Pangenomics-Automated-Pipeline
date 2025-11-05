import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from collections import Counter
import re
import random
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# UNIFIED STYLING CONFIGURATION - Matching Your Existing Aesthetics
# ==============================================================================

# Set consistent style matching your existing visualizations
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3

# Unified color palette - matching your existing scheme
COLORS = {
    'blue': '#3498db',
    'green': '#2ecc71',
    'red': '#e74c3c',
    'orange': '#f39c12',
    'purple': '#9b59b6',
    'teal': '#1abc9c',
    'darkblue': '#2c3e50',
    'lightblue': 'lightblue',
    'gray': '#95a5a6',
}

# Color schemes for different plot types
CLASS_COLORS = {
    'unique': COLORS['blue'],
    'accessory': COLORS['green'],
    'core': COLORS['red'],
    'other': COLORS['gray']
}

PIE_COLORS = [COLORS['blue'], COLORS['green'], COLORS['red'], COLORS['orange']]
CATEGORICAL_COLORS = [COLORS['blue'], COLORS['green'], COLORS['red'], COLORS['orange'], 
                      COLORS['purple'], COLORS['teal'], '#8B5A3C', '#47A025', 
                      '#D4A574', '#5E548E', '#BE95C4', '#3A506B']

print("="*80)
print("COMPLETE CD-HIT PANGENOME ANALYSIS AND VISUALIZATION")
print("="*80)

# ==============================================================================
# CONFIGURE DATA DIRECTORY
# ==============================================================================
if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    data_dir = 'Escherichia_coli'

print(f"\nData directory: {data_dir}\n")

# Create output directory for all figures
output_dir = os.path.join(data_dir, 'pangenome_figures')
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def save_figure(filename):
    """Save figure with consistent settings"""
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {filename}")
    plt.close()

def extract_cluster_num(x):
    """Extract cluster number from various formats"""
    if pd.isna(x):
        return None
    x_str = str(x)
    if 'Cluster' in x_str:
        return int(x_str.split()[-1])
    try:
        return int(x_str)
    except:
        return None

def extract_strain(seq_id):
    """Extract strain ID from sequence ID (e.g., 245894_CBEc5_03324 -> CBEc5)"""
    parts = seq_id.split("_")
    return parts[1] if len(parts) > 1 else seq_id

# ==============================================================================
# SECTION 1: LOAD ORIGINAL DATA FILES (YOUR EXISTING WORKFLOW)
# ==============================================================================
print("[SECTION 1] Loading original pangenome data files...")

try:
    # Load pangenome data
    pangenome_path = os.path.join(data_dir, 'Escherichia_coli_pangenome.csv')
    pangenome = pd.read_csv(pangenome_path)
    pangenome.columns = [col.strip() for col in pangenome.columns]
    print(f"  ✓ Loaded pangenome file: {len(pangenome)} clusters")
    
    # Load distribution data
    distribution_path = os.path.join(data_dir, 'Escherichia_coli_distributions.csv')
    distribution = pd.read_csv(distribution_path)
    distribution.columns = [col.strip() for col in distribution.columns]
    print(f"  ✓ Loaded distribution file: {len(distribution)} frequency bins")
    
    # Parse cluster file
    cluster_path = os.path.join(data_dir, 'Escherichia_coli.fasta.clstr')
    clusters_original = []
    current_cluster = None
    cluster_sizes_original = {}
    
    with open(cluster_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>Cluster'):
                current_cluster = int(line.split()[1])
                cluster_sizes_original[current_cluster] = 0
            elif line and current_cluster is not None:
                cluster_sizes_original[current_cluster] += 1
                parts = line.split()
                seq_len = int(parts[1].replace('aa,', ''))
                identity = 100.0
                is_rep = '*' in line
                if 'at' in line and not is_rep:
                    try:
                        identity = float(line.split('at ')[-1].replace('%', ''))
                    except:
                        identity = 100.0
                clusters_original.append({
                    'cluster': current_cluster,
                    'sequence_length': seq_len,
                    'identity': identity,
                    'is_representative': is_rep
                })
    
    cluster_df = pd.DataFrame(clusters_original)
    print(f"  ✓ Parsed cluster file: {len(cluster_df)} sequences in {len(cluster_sizes_original)} clusters")
    
    # Data preparation
    cluster_col = pangenome.columns[0]
    pangenome['cluster_num'] = pangenome[cluster_col].apply(extract_cluster_num)
    pangenome['sequences_in_cluster'] = pangenome['cluster_num'].map(
        lambda x: cluster_sizes_original.get(x, 0) if x is not None else 0
    )
    
    total_clusters = len(pangenome)
    total_sequences = len(cluster_df)
    class_distribution = pangenome['gene_class'].value_counts()
    
    print(f"  ✓ Total clusters: {total_clusters}")
    print(f"  ✓ Total sequences: {total_sequences}")
    print(f"  ✓ Gene classes: {dict(class_distribution)}")
    
    has_original_data = True
    
except Exception as e:
    print(f"  ⚠ Original data files not found: {e}")
    print(f"  → Will only generate matrix-based visualizations")
    has_original_data = False

# ==============================================================================
# SECTION 2: LOAD MATRIX DATA (FOR NEW VISUALIZATIONS)
# ==============================================================================
print("\n[SECTION 2] Loading cluster matrix data for new visualizations...")

def parse_clstr(path):
    """Parse CD-HIT .clstr file for matrix analysis"""
    clusters = {}
    current = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                current = int(line.split()[1])
                clusters[current] = []
            else:
                m = re.search(r">(.*?)\.\.\.", line)
                if not m:
                    continue
                seq_id = m.group(1)
                clusters[current].append(seq_id)
    return clusters

def parse_clstr_for_stacked_bar(path):
    """Parse .clstr for stacked bar visualization"""
    reps = {}
    strain_map = {}
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                current = int(line.split()[1])
                strain_map[current] = set()
            else:
                m = re.match(r"\d+\s+(\d+)aa,\s+>([^.]+).*?(\*)?$", line)
                if m:
                    length = int(m.group(1))
                    seq_id = m.group(2)
                    strain = extract_strain(seq_id)
                    strain_map[current].add(strain)
                    if m.group(3) == '*':
                        reps[current] = (strain, length)
    return reps, strain_map

try:
    # Find the .clstr file
    if has_original_data:
        clstr_path = cluster_path
    else:
        # Search for any .clstr file
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".clstr"):
                    clstr_path = os.path.join(root, f)
                    break
    
    clusters = parse_clstr(clstr_path)
    print(f"  ✓ Parsed {len(clusters)} clusters for matrix analysis")
    
    # Build presence/absence matrix
    strain_set = sorted({extract_strain(s) for seqs in clusters.values() for s in seqs})
    presence_absence = pd.DataFrame(0, index=clusters.keys(), columns=strain_set)
    
    for cid, seqs in clusters.items():
        strains = {extract_strain(s) for s in seqs}
        for strain in strains:
            presence_absence.loc[cid, strain] = 1
    
    print(f"  ✓ Built presence/absence matrix: {len(presence_absence)} clusters × {len(strain_set)} strains")
    print(f"  ✓ Strains: {', '.join(strain_set)}")
    
    has_matrix_data = True
    
except Exception as e:
    print(f"  ⚠ Could not build presence/absence matrix: {e}")
    has_matrix_data = False

# ==============================================================================
# FIGURE 1: PANGENOME OVERVIEW (YOUR ORIGINAL)
# ==============================================================================
if has_original_data:
    print("\n[FIGURE 1] Generating Pangenome Overview...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # 1.1 Gene class distribution (Pie)
    ax1 = fig.add_subplot(gs[0, 0])
    wedges, texts, autotexts = ax1.pie(
        class_distribution.values, 
        labels=class_distribution.index,
        autopct='%1.1f%%',
        colors=PIE_COLORS[:len(class_distribution)],
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    ax1.set_title('A. Gene Class Distribution', fontsize=13, weight='bold', pad=15)
    
    # 1.2 Cluster size histogram
    ax2 = fig.add_subplot(gs[0, 1])
    genome_counts = pangenome['Number of genomes']
    bins = min(50, int(genome_counts.max() / 5)) if genome_counts.max() > 50 else 20
    n, bins_edge, patches = ax2.hist(genome_counts, bins=bins, color=COLORS['blue'], 
                                edgecolor='black', alpha=0.7, linewidth=1.2)
    ax2.axvline(genome_counts.median(), color=COLORS['red'], linestyle='--', 
                linewidth=2, label=f'Median: {genome_counts.median():.0f}')
    ax2.axvline(genome_counts.mean(), color=COLORS['orange'], linestyle='--', 
                linewidth=2, label=f'Mean: {genome_counts.mean():.1f}')
    ax2.set_xlabel('Number of Genomes per Cluster', fontsize=11, weight='bold')
    ax2.set_ylabel('Number of Clusters', fontsize=11, weight='bold')
    ax2.set_title('B. Cluster Size Distribution', fontsize=13, weight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # 1.3 Box plot by gene class
    ax3 = fig.add_subplot(gs[0, 2])
    gene_classes = pangenome['gene_class'].unique()
    data_to_plot = [pangenome[pangenome['gene_class'] == gc]['Number of genomes'].values 
                    for gc in gene_classes]
    bp = ax3.boxplot(data_to_plot, labels=gene_classes, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2),
                     boxprops=dict(facecolor=COLORS['lightblue'], alpha=0.7))
    ax3.set_ylabel('Number of Genomes', fontsize=11, weight='bold')
    ax3.set_xlabel('Gene Class', fontsize=11, weight='bold')
    ax3.set_title('C. Genome Distribution by Class', fontsize=13, weight='bold', pad=15)
    ax3.grid(axis='y', alpha=0.3)
    
    # 1.4 Top 15 largest clusters
    ax4 = fig.add_subplot(gs[1, :])
    top15 = pangenome.nlargest(15, 'Number of genomes')
    cluster_names = [f"Cluster {extract_cluster_num(x)}" for x in top15[cluster_col]]
    bar_colors = [CLASS_COLORS.get(c, COLORS['gray']) for c in top15['gene_class']]
    bars = ax4.barh(range(len(top15)), top15['Number of genomes'], 
                    color=bar_colors, edgecolor='black', linewidth=1.2)
    ax4.set_yticks(range(len(top15)))
    ax4.set_yticklabels(cluster_names, fontsize=10)
    ax4.set_xlabel('Number of Genomes', fontsize=11, weight='bold')
    ax4.set_title('D. Top 15 Largest Gene Clusters', fontsize=13, weight='bold', pad=15)
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
    
    # Legend for gene classes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CLASS_COLORS[k], edgecolor='black', 
                            label=k.capitalize()) for k in CLASS_COLORS.keys() if k != 'other']
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # 1.5 Sequences per cluster
    ax5 = fig.add_subplot(gs[2, 0])
    seq_per_cluster = pangenome['sequences_in_cluster']
    ax5.hist(seq_per_cluster, bins=30, color=COLORS['purple'], 
             edgecolor='black', alpha=0.7, linewidth=1.2)
    ax5.axvline(seq_per_cluster.median(), color=COLORS['red'], linestyle='--',
                linewidth=2, label=f'Median: {seq_per_cluster.median():.0f}')
    ax5.set_xlabel('Sequences per Cluster', fontsize=11, weight='bold')
    ax5.set_ylabel('Number of Clusters', fontsize=11, weight='bold')
    ax5.set_title('E. Cluster Sequence Counts', fontsize=13, weight='bold', pad=15)
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    
    # 1.6 Statistics box
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('off')
    stats_text = f"""
    PANGENOME STATISTICS
    {'='*50}
    
    Total Clusters:           {total_clusters:,}
    Total Sequences:          {total_sequences:,}
    Average Sequences/Cluster: {total_sequences/total_clusters:.1f}
    
    Gene Classes:
    """
    for gene_class, count in class_distribution.items():
        pct = count / total_clusters * 100
        stats_text += f"    • {gene_class.capitalize()}: {count:,} ({pct:.1f}%)\n"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    save_figure('01_pangenome_overview.png')

# ==============================================================================
# FIGURE 2: GENE FREQUENCY ANALYSIS (YOUR ORIGINAL)
# ==============================================================================
if has_original_data:
    print("\n[FIGURE 2] Generating Gene Frequency Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Gene Frequency and Distribution Analysis', fontsize=16, weight='bold')
    
    # 2.1 Frequency distribution
    ax1 = axes[0, 0]
    freq_data = distribution['var_freq'].values
    count_data = distribution['var_count'].values
    
    ax1.bar(freq_data, count_data, color=COLORS['blue'], 
            edgecolor='black', alpha=0.7, linewidth=1.2, width=0.8)
    ax1.set_xlabel('Gene Frequency (# of Genomes)', fontsize=11, weight='bold')
    ax1.set_ylabel('Number of Genes', fontsize=11, weight='bold')
    ax1.set_title('A. Gene Frequency Distribution', fontsize=13, weight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2.2 Cumulative distribution
    ax2 = axes[0, 1]
    cumulative = distribution['var_cum_count'].values
    ax2.plot(freq_data, cumulative, marker='o', linewidth=2.5, 
             markersize=6, color=COLORS['green'], markerfacecolor=COLORS['green'],
             markeredgecolor='black', markeredgewidth=1)
    ax2.fill_between(freq_data, cumulative, alpha=0.3, color=COLORS['green'])
    ax2.set_xlabel('Gene Frequency', fontsize=11, weight='bold')
    ax2.set_ylabel('Cumulative Gene Count', fontsize=11, weight='bold')
    ax2.set_title('B. Cumulative Distribution', fontsize=13, weight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    
    # 2.3 Relative frequency
    ax3 = axes[1, 0]
    total_genes = distribution['var_cum_count'].iloc[-1]
    relative_freq = (distribution['var_count'] / total_genes) * 100
    
    ax3.bar(freq_data, relative_freq, color=COLORS['teal'], 
            edgecolor='black', alpha=0.7, linewidth=1.2, width=0.8)
    ax3.set_xlabel('Gene Frequency', fontsize=11, weight='bold')
    ax3.set_ylabel('Percentage of Total Genes', fontsize=11, weight='bold')
    ax3.set_title('C. Relative Frequency Distribution', fontsize=13, weight='bold', pad=15)
    ax3.grid(axis='y', alpha=0.3)
    
    # Highlight singletons
    singleton_pct = relative_freq.iloc[0]
    ax3.bar(1, singleton_pct, color=COLORS['red'], edgecolor='black', 
            linewidth=1.2, alpha=0.7, label=f'Singletons: {singleton_pct:.1f}%')
    ax3.legend(fontsize=10)
    
    # 2.4 Power law analysis
    ax4 = axes[1, 1]
    x = distribution['var_freq'].values
    y = distribution['var_count'].values
    mask = (x > 0) & (y > 0)
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    ax4.scatter(x_filtered, y_filtered, s=120, alpha=0.7, color=COLORS['purple'], 
                edgecolor='black', linewidth=1.5, zorder=3)
    ax4.set_xlabel('Gene Frequency (log scale)', fontsize=11, weight='bold')
    ax4.set_ylabel('Gene Count (log scale)', fontsize=11, weight='bold')
    ax4.set_title('D. Power Law Distribution', fontsize=13, weight='bold', pad=15)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    # Fit power law
    log_x = np.log10(x_filtered)
    log_y = np.log10(y_filtered)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fit_x = np.logspace(np.log10(x_filtered.min()), np.log10(x_filtered.max()), 100)
    fit_y = 10**intercept * fit_x**slope
    ax4.plot(fit_x, fit_y, 'r--', linewidth=2.5, 
             label=f'Power law fit\nSlope: {slope:.2f}', zorder=2)
    ax4.legend(fontsize=10, loc='upper right')
    ax4.grid(True, alpha=0.3, which='both')
    
    save_figure('02_gene_frequency_analysis.png')

# ==============================================================================
# FIGURE 3: SEQUENCE ANALYSIS (YOUR ORIGINAL)
# ==============================================================================
if has_original_data:
    print("\n[FIGURE 3] Generating Sequence Identity Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sequence Identity and Length Analysis', fontsize=16, weight='bold')
    
    # 3.1 Sequence identity distribution
    ax1 = axes[0, 0]
    identity_data = cluster_df[cluster_df['is_representative'] == False]['identity']
    if len(identity_data) > 0:
        ax1.hist(identity_data, bins=30, color=COLORS['teal'], 
                 edgecolor='black', alpha=0.7, linewidth=1.2)
        ax1.axvline(identity_data.mean(), color=COLORS['red'], linestyle='--',
                    linewidth=2.5, label=f'Mean: {identity_data.mean():.2f}%')
        ax1.axvline(identity_data.median(), color=COLORS['orange'], linestyle='--',
                    linewidth=2.5, label=f'Median: {identity_data.median():.2f}%')
        ax1.legend(fontsize=10)
    ax1.set_xlabel('Sequence Identity (%)', fontsize=11, weight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, weight='bold')
    ax1.set_title('A. Sequence Identity Distribution', fontsize=13, weight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # 3.2 Sequence length distribution
    ax2 = axes[0, 1]
    length_data = cluster_df['sequence_length']
    ax2.hist(length_data, bins=40, color=COLORS['orange'], 
             edgecolor='black', alpha=0.7, linewidth=1.2)
    ax2.axvline(length_data.mean(), color=COLORS['red'], linestyle='--',
                linewidth=2.5, label=f'Mean: {length_data.mean():.0f} aa')
    ax2.set_xlabel('Sequence Length (amino acids)', fontsize=11, weight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, weight='bold')
    ax2.set_title('B. Sequence Length Distribution', fontsize=13, weight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3.3 Identity categories
    ax3 = axes[1, 0]
    if len(identity_data) > 0:
        identity_bins = pd.cut(identity_data, bins=[0, 80, 90, 95, 100], 
                               labels=['<80%', '80-90%', '90-95%', '95-100%'])
        identity_counts = identity_bins.value_counts().sort_index()
        bars = ax3.bar(range(len(identity_counts)), identity_counts.values,
                       color=[COLORS['red'], COLORS['orange'], COLORS['blue'], COLORS['green']],
                       edgecolor='black', linewidth=1.2, alpha=0.7)
        ax3.set_xticks(range(len(identity_counts)))
        ax3.set_xticklabels(identity_counts.index, rotation=0)
        for bar, val in zip(bars, identity_counts.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                     f'{val}', ha='center', va='bottom', fontsize=9, weight='bold')
    ax3.set_ylabel('Number of Sequences', fontsize=11, weight='bold')
    ax3.set_xlabel('Identity Range', fontsize=11, weight='bold')
    ax3.set_title('C. Identity Categories', fontsize=13, weight='bold', pad=15)
    ax3.grid(axis='y', alpha=0.3)
    
    # 3.4 Sequences per cluster
    ax4 = axes[1, 1]
    cluster_sequence_counts = cluster_df.groupby('cluster').size()
    ax4.hist(cluster_sequence_counts, bins=30, color=COLORS['purple'],
             edgecolor='black', alpha=0.7, linewidth=1.2)
    ax4.set_xlabel('Sequences per Cluster', fontsize=11, weight='bold')
    ax4.set_ylabel('Number of Clusters', fontsize=11, weight='bold')
    ax4.set_title('D. Sequences per Cluster', fontsize=13, weight='bold', pad=15)
    ax4.axvline(cluster_sequence_counts.median(), color='red', linestyle='--',
                linewidth=2.5, label=f'Median: {cluster_sequence_counts.median():.0f}')
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    save_figure('03_sequence_analysis.png')

# ==============================================================================
# FIGURE 4: PRESENCE/ABSENCE MATRIX (NEW)
# ==============================================================================
if has_matrix_data:
    print("\n[FIGURE 4] Generating Presence/Absence Matrix...")
    
    N_MATRIX = 30
    subset = presence_absence.iloc[:N_MATRIX]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create matrix with matching color scheme
    im = ax.imshow(subset, cmap='RdYlBu_r', aspect='auto', 
                   interpolation='nearest', vmin=0, vmax=1)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Gene Presence', rotation=270, labelpad=20, weight='bold', fontsize=11)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Absent', 'Present'])
    
    # Format axes
    ax.set_xticks(np.arange(len(subset.columns)))
    ax.set_yticks(np.arange(len(subset.index)))
    ax.set_xticklabels(subset.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(subset.index, fontsize=10)
    
    # Add gridlines
    ax.set_xticks(np.arange(len(subset.columns))-0.5, minor=True)
    ax.set_yticks(np.arange(len(subset.index))-0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    
    # Labels
    ax.set_xlabel('Strain', weight='bold', fontsize=12)
    ax.set_ylabel('Cluster ID', weight='bold', fontsize=12)
    ax.set_title(f'Gene Cluster Presence/Absence Matrix (First {N_MATRIX} Clusters)', 
                 pad=20, weight='bold', fontsize=14)
    
    # Border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    save_figure('04_presence_absence_matrix.png')

# ==============================================================================
# ==============================================================================
# FIGURE 5: CORE AND PAN-GENOME ACCUMULATION CURVES (CLEANEST VERSION)
# ==============================================================================
if has_matrix_data:
    print("\n[FIGURE 5] Generating Core and Pan-Genome Curves...")
    
    def core_pan_curve(presence_absence, n_perm=100):
        strains = list(presence_absence.columns)
        n_strains = len(strains)
        core_counts = np.zeros((n_perm, n_strains))
        pan_counts = np.zeros((n_perm, n_strains))
        
        for p in range(n_perm):
            order = random.sample(strains, len(strains))
            current_pan = set()
            current_core = set(presence_absence.index)
            
            for i, strain in enumerate(order):
                strain_clusters = set(presence_absence.index[presence_absence[strain] == 1])
                current_pan |= strain_clusters
                pan_counts[p, i] = len(current_pan)
                current_core &= strain_clusters
                core_counts[p, i] = len(current_core)
        
        mean_pan = pan_counts.mean(axis=0)
        mean_core = core_counts.mean(axis=0)
        std_pan = pan_counts.std(axis=0)
        std_core = core_counts.std(axis=0)
        
        return mean_core, mean_pan, std_core, std_pan
    
    mean_core, mean_pan, std_core, std_pan = core_pan_curve(presence_absence, n_perm=100)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(1, len(mean_core) + 1)
    
    # Plot pan-genome
    ax.plot(x, mean_pan, marker='o', linewidth=3, markersize=8, 
            color=COLORS['blue'], label='Pan-genome', zorder=3)
    ax.fill_between(x, mean_pan - std_pan, mean_pan + std_pan, 
                    alpha=0.2, color=COLORS['blue'], zorder=1)
    
    # Plot core-genome
    ax.plot(x, mean_core, marker='s', linewidth=3, markersize=8, 
            color=COLORS['red'], label='Core-genome', zorder=3)
    ax.fill_between(x, mean_core - std_core, mean_core + std_core, 
                    alpha=0.2, color=COLORS['red'], zorder=1)
    
    # Formatting
    ax.set_xlabel('Number of Genomes', weight='bold', fontsize=12)
    ax.set_ylabel('Number of Gene Clusters', weight='bold', fontsize=12)
    ax.set_title('Core and Pan-Genome Accumulation Curves', 
                 pad=20, weight='bold', fontsize=14)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    legend = ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, 
                       fontsize=11, framealpha=0.95)
    legend.get_frame().set_linewidth(1.5)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    # BEST: Use nice round numbers for x-axis ticks
    n_genomes = len(x)
    
    # Determine appropriate step size for round numbers
    if n_genomes <= 10:
        step = 1  # 1, 2, 3, 4, 5...
    elif n_genomes <= 20:
        step = 2  # 2, 4, 6, 8, 10...
    elif n_genomes <= 50:
        step = 5  # 5, 10, 15, 20, 25...
    else:
        step = 10  # 10, 20, 30, 40, 50...
    
    # Create nice round tick positions
    tick_positions = list(range(0, n_genomes + 1, step))
    
    # Remove 0 if present and ensure we have good coverage
    if 0 in tick_positions:
        tick_positions.remove(0)
    
    # Make sure we don't exceed the data range
    tick_positions = [t for t in tick_positions if t <= n_genomes]
    
    # Optionally add the max value if it's not too close to last tick
    if tick_positions[-1] != n_genomes and (n_genomes - tick_positions[-1]) >= step / 2:
        tick_positions.append(n_genomes)
    
    ax.set_xticks(tick_positions)
    ax.set_xlim(0, n_genomes + 1)  # Start from 0 for cleaner look
    
    save_figure('05_core_pan_genome_curves.png')
# ==============================================================================
# FIGURE 6: STACKED BAR CHART (NEW)
# ==============================================================================
if has_matrix_data:
    print("\n[FIGURE 6] Generating Stacked Bar Chart...")
    
    N_STACKED = 25
    reps, strain_map = parse_clstr_for_stacked_bar(clstr_path)
    first_clusters = list(reps.keys())[:N_STACKED]
    
    all_strains = sorted({s for strains in strain_map.values() for s in strains})
    df = pd.DataFrame(0, index=first_clusters, columns=all_strains, dtype=float)
    
    for cid in first_clusters:
        if cid in reps:
            rep_strain, rep_len = reps[cid]
            strains = strain_map[cid]
            for s in strains:
                df.loc[cid, s] = rep_len / len(strains)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    df.plot(kind='bar', stacked=True, ax=ax, 
            color=CATEGORICAL_COLORS[:len(all_strains)],
            width=0.85, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Cluster ID', weight='bold', fontsize=12)
    ax.set_ylabel('Representative Sequence Length (aa)', weight='bold', fontsize=12)
    ax.set_title(f'Gene Cluster Composition by Strain (First {N_STACKED} Clusters)', 
                 pad=20, weight='bold', fontsize=14)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    legend = ax.legend(title='Strain', bbox_to_anchor=(1.02, 1), loc='upper left',
                       frameon=True, fancybox=True, shadow=True, 
                       title_fontsize=11, fontsize=10, framealpha=0.95)
    legend.get_frame().set_linewidth(1.5)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    save_figure('06_stacked_bar_clusters.png')

# ==============================================================================
# FIGURE 7: CLUSTERED STRAIN SIMILARITY HEATMAP (FINAL VERSION)
# ==============================================================================
if has_matrix_data:
    print("\n[FIGURE 7] Generating Clustered Strain Similarity Heatmap...")

    # Calculate strain similarity matrix
    strain_similarity = presence_absence.T.dot(presence_absence)

    # Create figure with appropriate size for many strains
    n_strains = len(strain_set)
    fig_size = max(14, n_strains * 0.25)  # Scale figure size with number of strains

    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    # Convert similarity to distance for clustering
    max_similarity = strain_similarity.values.max()
    distance_matrix = max_similarity - strain_similarity.values
    np.fill_diagonal(distance_matrix, 0)

    # Perform hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')

    # Get the order from clustering
    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']

    # Reorder the similarity matrix
    strain_similarity_clustered = strain_similarity.iloc[order, order]
    strain_labels_ordered = [strain_set[i] for i in order]

    # Create clustered heatmap
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(strain_similarity_clustered, cmap='viridis',
                   aspect='auto', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Number of Shared Gene Clusters', rotation=270,
                   labelpad=25, weight='bold', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Axis labels
    ax.set_xticks(np.arange(len(strain_labels_ordered)))
    ax.set_yticks(np.arange(len(strain_labels_ordered)))
    label_fontsize = max(6, 12 - n_strains // 10)
    ax.set_xticklabels(strain_labels_ordered, rotation=90, ha='right', fontsize=label_fontsize)
    ax.set_yticklabels(strain_labels_ordered, fontsize=label_fontsize)

    # Labels and title
    ax.set_xlabel('Strain', weight='bold', fontsize=14)
    ax.set_ylabel('Strain', weight='bold', fontsize=14)
    ax.set_title('Clustered Pairwise Strain Similarity Based on Shared Gene Clusters',
                 pad=20, weight='bold', fontsize=16)

    # Grid and border styling
    ax.set_xticks(np.arange(len(strain_labels_ordered)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(strain_labels_ordered)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    plt.tight_layout()

    # Save final clustered heatmap
    save_figure('07_strain_similarity_heatmap_clustered.png')


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("✅ VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nAll figures saved to: {output_dir}/")
print("\nGenerated files:")

if has_original_data:
    print("  ORIGINAL ANALYSES:")
    print("    01_pangenome_overview.png")
    print("    02_gene_frequency_analysis.png")
    print("    03_sequence_analysis.png")

if has_matrix_data:
    print("  NEW MATRIX ANALYSES:")
    print("    04_presence_absence_matrix.png")
    print("    05_core_pan_genome_curves.png")
    print("    06_stacked_bar_clusters.png")
    print("    07_strain_similarity_heatmap.png")

print("\n" + "="*80)
print("All visualizations use unified aesthetics:")
print("  ✓ Consistent color palette")
print("  ✓ Matching fonts and sizing")
print("  ✓ Publication-quality (300 DPI)")
print("  ✓ Coherent styling across all figures")
print("="*80)