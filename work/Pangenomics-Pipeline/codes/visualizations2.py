import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

print("="*80)
print("CD-HIT PANGENOME ANALYSIS AND VISUALIZATION")
print("="*80)

# ============================================================================
# CONFIGURE DATA DIRECTORY
# ============================================================================
# You can pass directory as command line argument or set it here
if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    # CHANGE THIS PATH to your data directory
    data_dir = 'Escherichia_coli'

print(f"\nData directory: {data_dir}\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("[1/5] Loading data files...")

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
    clusters = []
    current_cluster = None
    cluster_sizes = {}
    
    with open(cluster_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>Cluster'):
                current_cluster = int(line.split()[1])
                cluster_sizes[current_cluster] = 0
            elif line and current_cluster is not None:
                cluster_sizes[current_cluster] += 1
                parts = line.split()
                seq_len = int(parts[1].replace('aa,', ''))
                identity = 100.0
                is_rep = '*' in line
                if 'at' in line and not is_rep:
                    try:
                        identity = float(line.split('at ')[-1].replace('%', ''))
                    except:
                        identity = 100.0
                clusters.append({
                    'cluster': current_cluster,
                    'sequence_length': seq_len,
                    'identity': identity,
                    'is_representative': is_rep
                })
    
    cluster_df = pd.DataFrame(clusters)
    print(f"  ✓ Parsed cluster file: {len(cluster_df)} sequences in {len(cluster_sizes)} clusters")
    
except Exception as e:
    print(f"  ✗ Error loading data: {e}")
    print(f"\nMake sure these files exist in: {data_dir}")
    print("  - Escherichia_coli_pangenome.csv")
    print("  - Escherichia_coli_distributions.csv")
    print("  - Escherichia_coli.fasta.clstr")
    exit(1)

# ============================================================================
# DATA PREPARATION
# ============================================================================
print("\n[2/5] Preparing data for analysis...")

# Get the cluster column name (it might be unnamed or have a specific name)
cluster_col = pangenome.columns[0]

# Extract cluster numbers properly
def extract_cluster_num(x):
    if pd.isna(x):
        return None
    x_str = str(x)
    if 'Cluster' in x_str:
        return int(x_str.split()[-1])
    try:
        return int(x_str)
    except:
        return None

pangenome['cluster_num'] = pangenome[cluster_col].apply(extract_cluster_num)

# Add cluster sizes to pangenome dataframe
pangenome['sequences_in_cluster'] = pangenome['cluster_num'].map(
    lambda x: cluster_sizes.get(x, 0) if x is not None else 0
)

# Calculate key metrics
total_clusters = len(pangenome)
total_sequences = len(cluster_df)
class_distribution = pangenome['gene_class'].value_counts()

print(f"  ✓ Total clusters: {total_clusters}")
print(f"  ✓ Total sequences: {total_sequences}")
print(f"  ✓ Gene classes: {dict(class_distribution)}")

# ============================================================================
# FIGURE 1: PANGENOME OVERVIEW
# ============================================================================
print("\n[3/5] Generating Figure 1: Pangenome Overview...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1.1 Gene class distribution (Pie)
ax1 = fig.add_subplot(gs[0, 0])
colors_pie = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
wedges, texts, autotexts = ax1.pie(
    class_distribution.values, 
    labels=class_distribution.index,
    autopct='%1.1f%%',
    colors=colors_pie[:len(class_distribution)],
    startangle=90,
    textprops={'fontsize': 11, 'weight': 'bold'}
)
ax1.set_title('A. Gene Class Distribution', fontsize=13, weight='bold', pad=15)

# 1.2 Cluster size histogram
ax2 = fig.add_subplot(gs[0, 1])
genome_counts = pangenome['Number of genomes']
bins = min(50, int(genome_counts.max() / 5)) if genome_counts.max() > 50 else 20
n, bins_edge, patches = ax2.hist(genome_counts, bins=bins, color='#3498db', 
                            edgecolor='black', alpha=0.7, linewidth=1.2)
ax2.axvline(genome_counts.median(), color='#e74c3c', linestyle='--', 
            linewidth=2, label=f'Median: {genome_counts.median():.0f}')
ax2.axvline(genome_counts.mean(), color='#f39c12', linestyle='--', 
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
                 boxprops=dict(facecolor='lightblue', alpha=0.7))
ax3.set_ylabel('Number of Genomes', fontsize=11, weight='bold')
ax3.set_xlabel('Gene Class', fontsize=11, weight='bold')
ax3.set_title('C. Genome Distribution by Class', fontsize=13, weight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3)

# 1.4 Top 15 largest clusters
ax4 = fig.add_subplot(gs[1, :])
top15 = pangenome.nlargest(15, 'Number of genomes')
cluster_names = [f"Cluster {extract_cluster_num(x)}" for x in top15[cluster_col]]
colors_map = {'unique': '#3498db', 'accessory': '#2ecc71', 'core': '#e74c3c'}
bar_colors = [colors_map.get(c, '#95a5a6') for c in top15['gene_class']]
bars = ax4.barh(range(len(top15)), top15['Number of genomes'], 
                color=bar_colors, edgecolor='black', linewidth=1.2)
ax4.set_yticks(range(len(top15)))
ax4.set_yticklabels(cluster_names, fontsize=10)
ax4.set_xlabel('Number of Genomes', fontsize=11, weight='bold')
ax4.set_title('D. Top 15 Largest Gene Clusters', fontsize=13, weight='bold', pad=15)
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, top15['Number of genomes'])):
    ax4.text(val + 1, i, str(val), va='center', fontsize=9, weight='bold')

# 1.5 Sequences per cluster distribution
ax5 = fig.add_subplot(gs[2, 0])
seq_per_cluster = pangenome['sequences_in_cluster']
seq_per_cluster_clean = seq_per_cluster[seq_per_cluster > 0]
ax5.hist(seq_per_cluster_clean, bins=30, color='#9b59b6', 
         edgecolor='black', alpha=0.7, linewidth=1.2)
ax5.set_xlabel('Sequences per Cluster', fontsize=11, weight='bold')
ax5.set_ylabel('Frequency', fontsize=11, weight='bold')
ax5.set_title('E. Cluster Sequence Counts', fontsize=13, weight='bold', pad=15)
if len(seq_per_cluster_clean) > 0:
    ax5.axvline(seq_per_cluster_clean.median(), color='red', linestyle='--', 
                linewidth=2, label=f'Median: {seq_per_cluster_clean.median():.0f}')
    ax5.legend(fontsize=10)
ax5.grid(axis='y', alpha=0.3)

# 1.6 Summary statistics
ax6 = fig.add_subplot(gs[2, 1:])
ax6.axis('off')
summary_data = []
for gc in gene_classes:
    subset = pangenome[pangenome['gene_class'] == gc]['Number of genomes']
    summary_data.append([
        gc,
        len(subset),
        f"{subset.mean():.1f}",
        f"{subset.median():.0f}",
        subset.min(),
        subset.max(),
        f"{len(subset)/len(pangenome)*100:.1f}%"
    ])

table = ax6.table(
    cellText=summary_data,
    colLabels=['Class', 'Count', 'Mean', 'Median', 'Min', 'Max', '% Total'],
    cellLoc='center',
    loc='center',
    colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)
for i in range(7):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')
for i in range(1, len(summary_data) + 1):
    for j in range(7):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
ax6.set_title('F. Summary Statistics by Gene Class', fontsize=13, weight='bold', pad=20)

plt.suptitle('CD-HIT Pangenome Overview', fontsize=16, weight='bold', y=0.995)
output_path = os.path.join(data_dir, '1_pangenome_overview.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")

# Print interpretation
print("\n" + "="*80)
print("INTERPRETATION - FIGURE 1: PANGENOME OVERVIEW")
print("="*80)
print(f"\n📊 Dataset Summary:")
print(f"  • Total gene clusters: {total_clusters}")
print(f"  • Total sequences: {total_sequences}")
print(f"  • Average sequences per cluster: {total_sequences/total_clusters:.1f}")

print(f"\n🧬 Gene Class Breakdown:")
for gc, count in class_distribution.items():
    pct = count/total_clusters*100
    print(f"  • {gc.capitalize()}: {count} clusters ({pct:.1f}%)")

print(f"\n📈 Cluster Size Statistics:")
print(f"  • Median: {genome_counts.median():.0f} genomes")
print(f"  • Mean: {genome_counts.mean():.1f} genomes")
print(f"  • Range: {genome_counts.min()} - {genome_counts.max()} genomes")

print(f"\n💡 Key Insights:")
unique_pct = class_distribution.get('unique', 0) / total_clusters * 100
if unique_pct > 60:
    print(f"  → HIGH unique gene content ({unique_pct:.1f}%) suggests:")
    print(f"     • High genetic diversity between strains")
    print(f"     • Open pangenome with strain-specific adaptations")
    print(f"     • Potential for discovering novel genes")
elif unique_pct < 30:
    print(f"  → LOW unique gene content ({unique_pct:.1f}%) suggests:")
    print(f"     • Closely related strains")
    print(f"     • Closed pangenome with limited diversity")
    print(f"     • Well-conserved core genome")
else:
    print(f"  → MODERATE unique gene content ({unique_pct:.1f}%) suggests:")
    print(f"     • Moderate genetic diversity")
    print(f"     • Balance between core and variable genes")

# ============================================================================
# FIGURE 2: GENE FREQUENCY ANALYSIS
# ============================================================================
print("\n[4/5] Generating Figure 2: Gene Frequency Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Gene Frequency Distribution Analysis', fontsize=16, weight='bold')

# 2.1 Gene frequency bar plot (log scale)
ax1 = axes[0, 0]
bars = ax1.bar(distribution['var_freq'], distribution['var_count'], 
               color='#3498db', edgecolor='black', alpha=0.7, linewidth=1.2)
ax1.set_xlabel('Gene Frequency (# of genomes)', fontsize=11, weight='bold')
ax1.set_ylabel('Number of Genes (log scale)', fontsize=11, weight='bold')
ax1.set_title('A. Gene Frequency Distribution', fontsize=13, weight='bold', pad=15)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3, which='both')
for i in range(min(5, len(distribution))):
    height = distribution['var_count'].iloc[i]
    ax1.text(distribution['var_freq'].iloc[i], height * 1.1, 
             f"{height}", ha='center', va='bottom', fontsize=9, weight='bold')

# 2.2 Cumulative gene accumulation
ax2 = axes[0, 1]
ax2.plot(distribution['var_freq'], distribution['var_cum_count'], 
         marker='o', linewidth=2.5, markersize=8, color='#e74c3c', 
         markerfacecolor='white', markeredgewidth=2)
ax2.fill_between(distribution['var_freq'], distribution['var_cum_count'], 
                  alpha=0.3, color='#e74c3c')
ax2.set_xlabel('Gene Frequency', fontsize=11, weight='bold')
ax2.set_ylabel('Cumulative Gene Count', fontsize=11, weight='bold')
ax2.set_title('B. Cumulative Gene Accumulation', fontsize=13, weight='bold', pad=15)
ax2.grid(True, alpha=0.3)
final_count = distribution['var_cum_count'].iloc[-1]
ax2.annotate(f'Total: {final_count}', 
             xy=(distribution['var_freq'].iloc[-1], final_count),
             xytext=(10, -20), textcoords='offset points',
             fontsize=10, weight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# 2.3 Relative frequency distribution
ax3 = axes[1, 0]
total_genes = distribution['var_count'].sum()
relative_freq = (distribution['var_count'] / total_genes * 100)
bars = ax3.bar(distribution['var_freq'], relative_freq, 
               color='#2ecc71', edgecolor='black', alpha=0.7, linewidth=1.2)
ax3.set_xlabel('Gene Frequency', fontsize=11, weight='bold')
ax3.set_ylabel('Percentage of Total Genes (%)', fontsize=11, weight='bold')
ax3.set_title('C. Relative Gene Frequency', fontsize=13, weight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3)
singleton_pct = relative_freq.iloc[0]
ax3.bar(1, singleton_pct, color='#e74c3c', edgecolor='black', 
        linewidth=1.2, alpha=0.7, label=f'Singletons: {singleton_pct:.1f}%')
ax3.legend(fontsize=10)

# 2.4 Power law analysis
ax4 = axes[1, 1]
x = distribution['var_freq'].values
y = distribution['var_count'].values
mask = (x > 0) & (y > 0)
x_filtered = x[mask]
y_filtered = y[mask]

ax4.scatter(x_filtered, y_filtered, s=120, alpha=0.7, color='#9b59b6', 
            edgecolor='black', linewidth=1.5, zorder=3)
ax4.set_xlabel('Gene Frequency (log scale)', fontsize=11, weight='bold')
ax4.set_ylabel('Gene Count (log scale)', fontsize=11, weight='bold')
ax4.set_title('D. Power Law Distribution', fontsize=13, weight='bold', pad=15)
ax4.set_xscale('log')
ax4.set_yscale('log')

log_x = np.log10(x_filtered)
log_y = np.log10(y_filtered)
slope, intercept = np.polyfit(log_x, log_y, 1)
fit_x = np.logspace(np.log10(x_filtered.min()), np.log10(x_filtered.max()), 100)
fit_y = 10**intercept * fit_x**slope
ax4.plot(fit_x, fit_y, 'r--', linewidth=2.5, 
         label=f'Power law fit\nSlope: {slope:.2f}', zorder=2)
ax4.legend(fontsize=10, loc='upper right')
ax4.grid(True, alpha=0.3, which='both')

plt.tight_layout()
output_path = os.path.join(data_dir, '2_gene_frequency_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")

# Print interpretation
print("\n" + "="*80)
print("INTERPRETATION - FIGURE 2: GENE FREQUENCY ANALYSIS")
print("="*80)

singleton_count = distribution[distribution['var_freq'] == 1]['var_count'].values[0]
total_genes_dist = distribution['var_cum_count'].iloc[-1]
singleton_pct = singleton_count / total_genes_dist * 100

print(f"\n📊 Gene Frequency Statistics:")
print(f"  • Total unique genes: {total_genes_dist:,}")
print(f"  • Singleton genes (freq=1): {singleton_count:,} ({singleton_pct:.1f}%)")
print(f"  • Most common frequency: {distribution.loc[distribution['var_count'].idxmax(), 'var_freq']}")
print(f"  • Power law slope: {slope:.2f}")

print(f"\n💡 Biological Interpretation:")
if singleton_pct > 50:
    print(f"  → VERY HIGH singleton content ({singleton_pct:.1f}%):")
    print(f"     • Extremely diverse pangenome")
    print(f"     • Each genome contributes many unique genes")
    print(f"     • Open pangenome - may never saturate")
elif singleton_pct > 30:
    print(f"  → HIGH singleton content ({singleton_pct:.1f}%):")
    print(f"     • Diverse accessory genome")
    print(f"     • Significant strain-specific functions")
elif singleton_pct < 20:
    print(f"  → LOW singleton content ({singleton_pct:.1f}%):")
    print(f"     • Conserved pangenome structure")
    print(f"     • Shared gene content across strains")
else:
    print(f"  → MODERATE singleton content ({singleton_pct:.1f}%):")
    print(f"     • Balanced pangenome")

print(f"\n📉 Power Law Analysis:")
if slope < -1.5:
    print(f"  → Steep slope ({slope:.2f}): Typical bacterial pangenome")
elif slope > -0.5:
    print(f"  → Shallow slope ({slope:.2f}): Unusual distribution")
else:
    print(f"  → Moderate slope ({slope:.2f}): Standard pattern")

# ============================================================================
# FIGURE 3: SEQUENCE ANALYSIS
# ============================================================================
print("\n[5/5] Generating Figure 3: Sequence Identity Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Sequence Identity and Length Analysis', fontsize=16, weight='bold')

# 3.1 Sequence identity distribution
ax1 = axes[0, 0]
identity_data = cluster_df[cluster_df['is_representative'] == False]['identity']
if len(identity_data) > 0:
    ax1.hist(identity_data, bins=30, color='#1abc9c', 
             edgecolor='black', alpha=0.7, linewidth=1.2)
    ax1.axvline(identity_data.mean(), color='#e74c3c', linestyle='--',
                linewidth=2.5, label=f'Mean: {identity_data.mean():.2f}%')
    ax1.axvline(identity_data.median(), color='#f39c12', linestyle='--',
                linewidth=2.5, label=f'Median: {identity_data.median():.2f}%')
    ax1.legend(fontsize=10)
ax1.set_xlabel('Sequence Identity (%)', fontsize=11, weight='bold')
ax1.set_ylabel('Frequency', fontsize=11, weight='bold')
ax1.set_title('A. Sequence Identity Distribution', fontsize=13, weight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3)

# 3.2 Sequence length distribution
ax2 = axes[0, 1]
length_data = cluster_df['sequence_length']
ax2.hist(length_data, bins=40, color='#e67e22', 
         edgecolor='black', alpha=0.7, linewidth=1.2)
ax2.axvline(length_data.mean(), color='#e74c3c', linestyle='--',
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
                   color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'],
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
ax4.hist(cluster_sequence_counts, bins=30, color='#9b59b6',
         edgecolor='black', alpha=0.7, linewidth=1.2)
ax4.set_xlabel('Sequences per Cluster', fontsize=11, weight='bold')
ax4.set_ylabel('Number of Clusters', fontsize=11, weight='bold')
ax4.set_title('D. Sequences per Cluster', fontsize=13, weight='bold', pad=15)
ax4.axvline(cluster_sequence_counts.median(), color='red', linestyle='--',
            linewidth=2.5, label=f'Median: {cluster_sequence_counts.median():.0f}')
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_path = os.path.join(data_dir, '3_sequence_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")

# Print final interpretation
print("\n" + "="*80)
print("INTERPRETATION - FIGURE 3: SEQUENCE ANALYSIS")
print("="*80)

if len(identity_data) > 0:
    print(f"\n📊 Sequence Identity Statistics:")
    print(f"  • Mean identity: {identity_data.mean():.2f}%")
    print(f"  • Median identity: {identity_data.median():.2f}%")
    print(f"  • Range: {identity_data.min():.2f}% - {identity_data.max():.2f}%")
    
    high_identity = (identity_data >= 95).sum()
    total_non_rep = len(identity_data)
    print(f"  • High identity (≥95%): {high_identity}/{total_non_rep} ({high_identity/total_non_rep*100:.1f}%)")
    
    mean_identity = identity_data.mean()
    print(f"\n💡 Clustering Quality:")
    if mean_identity >= 95:
        print(f"  → EXCELLENT ({mean_identity:.2f}%): Very tight, reliable clusters")
    elif mean_identity >= 85:
        print(f"  → GOOD ({mean_identity:.2f}%): Well-defined clusters")
    else:
        print(f"  → MODERATE ({mean_identity:.2f}%): Broader clusters")

print(f"\n📏 Sequence Length Statistics:")
print(f"  • Mean length: {length_data.mean():.0f} aa")
print(f"  • Median length: {length_data.median():.0f} aa")
print(f"  • Range: {length_data.min()} - {length_data.max()} aa")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll figures saved to: {data_dir}")
print("Files generated:")
print("  1. 1_pangenome_overview.png")
print("  2. 2_gene_frequency_analysis.png")
print("  3. 3_sequence_analysis.png")
print("\n" + "="*80)