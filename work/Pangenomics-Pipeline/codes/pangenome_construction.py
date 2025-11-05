import pandas as pd
import scipy.optimize
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import bisect
import sys

species = sys.argv[1]
total_genomes = int(sys.argv[2])  # Get total number of genomes as second argument

def construct_pangenome(species, total_genomes, save = True):
    var_counts = pd.read_csv(species+'/' +species+'_cluster_frequencies.csv', index_col = 0)
    var_freq = var_counts.groupby(by = 'Number of genomes', as_index = False).count()
    var_freq.columns = ['var_freq', 'var_count']

    def rescale(x, x_minmax = np.nan, back = False):
      if(back):
        return(x * (x_minmax[1] - x_minmax[0]) + x_minmax[0])
      else:
        if(type(x_minmax) == float):
          return((x - x.min())/(x.max() - x.min()))
        else:
          return((x - x_minmax[0])/(x_minmax[1] - x_minmax[0]))

    var_freq['var_norm_count'] = rescale(var_freq.var_count)
    var_freq.sort_values(by = 'var_freq')
    var_freq['var_cum_norm_count'] = var_freq.var_norm_count.cumsum()
    var_freq['var_cum_count'] = var_freq.var_count.cumsum()
    var_freq['var_norm_freq'] = rescale(var_freq.var_freq, [var_freq.var_freq.min(), var_freq.var_freq.max()])

    var_freq['var_log_freq'] = np.log10(var_freq.var_freq)
    var_freq['var_log_cum_count'] = np.log10(var_freq.var_cum_count)

    print("\n        Working on " + species)
    print("______________________________________________________")
     # + " with " + str(var_freq.var_freq.max()) + " genomes...")
    print()

    def cfd(x, c1, c2, a1, a2, k):
      return(k + (c1/(1-a1))*np.power(x,(1-a1)) - (c2/(1-a2))*np.power((np.max(x) + 1 - x),(1-a2)))

    params, cors = scipy.optimize.curve_fit(f = cfd,
                                            xdata = np.array(var_freq['var_freq']),
                                            ydata = np.array(var_freq['var_cum_norm_count']),
                                            p0 = (1,1,2,2,1), maxfev = 1000000, method = 'trf', bounds = (0, np.inf))

    print("Fitted parameters: " + str(params))

    def fd(x, c1, c2, a1, a2, N):
      return(c1*np.power(x,(-a1)) + c2*np.power((N + 1 - x),(-a2)))

    sol = scipy.optimize.minimize_scalar(fun = fd,
                                         bounds = (1,total_genomes), method = "bounded",
                                         args = tuple([params[0], params[1], params[2], params[3], total_genomes]))

    unique_t = 0.1*sol.x
    core_t = 0.9*total_genomes + 0.1*sol.x
    norm_unique_t = rescale(unique_t, [var_freq.var_freq.min(), var_freq.var_freq.max()])
    norm_core_t = rescale(core_t, [var_freq.var_freq.min(), var_freq.var_freq.max()])
    print("Unique Genes cutoff: " + str(norm_unique_t*100) + "% (" + str(unique_t) + ")")
    print("Core Genes cutoff: " + str(norm_core_t*100) + "% (" + str(core_t) + ")")

    core_filter = np.array(var_counts['Number of genomes'] > round(core_t))
    unique_filter = np.array(var_counts['Number of genomes'] < round(unique_t))
    accessory_filter = np.array([not (unique_filter[i] or core_filter[i]) for i in range(len(core_filter))])
    core_count = np.sum(core_filter)
    unique_count = np.sum(unique_filter)
    accessory_count = np.sum(accessory_filter)
    print()
    print("== Pangenome Division ==")
    print("Core: " + str(core_count) + " (" + str(100*core_count/var_counts.shape[0]) + "%)")
    print("Accessory: " + str(accessory_count) + " (" + str(100*accessory_count/var_counts.shape[0]) + "%)")
    print("Unique: " + str(unique_count) + " (" + str(100*unique_count/var_counts.shape[0]) + "%)")
    print("Total: " + str(var_counts.shape[0]))

    var_freq['cum_norm_y_hat'] = cfd(np.array(var_freq['var_freq']), *params)
    residuals = np.array(var_freq['var_cum_norm_count']) - var_freq['cum_norm_y_hat']
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.array(var_freq['var_cum_norm_count'])-np.mean(np.array(var_freq['var_cum_norm_count'])))**2)
    r_squared = 1 - (ss_res / ss_tot)

    var_freq['cum_y_hat'] = rescale(pd.Series([0] + var_freq['cum_norm_y_hat'].tolist()).diff()[1:].reset_index(drop=True), [var_freq.var_count.min(), var_freq.var_count.max()], back = True).cumsum()

    residuals = np.array(var_freq['var_cum_count']) - var_freq['cum_y_hat']
    mae = np.mean(np.abs(residuals))
    print()
    print("== Metrics ==")
    print("R-squared: " + str(r_squared))
    print("MAE: " + str(mae))

    var_counts.plot(bins = 60, kind = 'hist')
    plt.savefig(species+'/'+species+'_distribution_gene_frequency.png')

    var_freq.plot(x = 0, y = 3, kind = 'scatter')
    plt.savefig(species+'/'+species+"_distribution_cumulative_gene_frequency.png")
    
    expected_fits = {'Acinetobacter_baumannii': [0.407387191, 0.169881697, 1.270126148, 1.717954649, 2.497122657],
                     'Campylobacter_coli': [0.676514875, 0.665273974, 1.375463883, 2.208358671, 2.749117964],
                     'Campylobacter_jejuni': [0.596340459586115, 0.586999530820406, 1.27254790795425, 2.42652677069156, 3.0678391959799],
                     'Enterobacter_cloacae': [0.603123535422197, 0.62657091500199, 1.42140416511035, 2.92481761092922, 2.38249726125534],
                     'Enterococcus_faecium': [0.893210882649621, 0.473960249311455, 1.05165755262121, 1.0409210206294, 8.55013550126279],
                     'Escherichia_coli': [0.674574104923473, 0.042063844091932, 1.42188358501309, 1.40821892239716, 2.57303404386593],
                     'Klebsiella_pneumoniae': [0.709579674439859, 0.061014405255371, 1.46973335893235, 1.35129860861299, 2.45181507470237],
                     'Neisseria_gonorrhoeae': [0.468077124339678, 0.396119921697024, 1.15419619089709, 1.41731408271129, 3.86061834769387],
                     'Pseudomonas_aeruginosa': [0.719699185656992, 0.168927810772083, 1.55151393154238, 1.59296314979798, 2.25376921065999],
                     'Salmonella_enterica': [0.594484409413897, 0.185625489754542, 1.33245058993664, 1.71755736914495, 2.72343950806249],
                     'Staphylococcus_aureus': [0.380950319675536, 0.327360247842034, 1.2247794299588, 1.97580244663302, 2.85242255758538],
                     'Streptococcus_pneumoniae': [0.38095032, 0.327360248, 1.22477943, 1.975802447, 2.852422558]}
    
    inf_pts = {'Acinetobacter_baumannii': [805.3845231, 80.53845231, 999.4384523],
               'Campylobacter_coli': [209.6528748, 20.96528748, 263.0652875],
               'Campylobacter_jejuni': [389.238514100145, 38.9238514100145, 444.823851410015],
               'Enterobacter_cloacae': [86.0463210306286, 8.60463210306286, 102.204632103063],
               'Enterococcus_faecium': [97.2993740312014, 9.72993740312014, 161.82993740312],
               'Escherichia_coli': [1484.22016223274, 148.422016223274, 1921.42201622327],
               'Klebsiella_pneumoniae': [1274.8657341735, 127.48657341735, 1832.98657341735],
               'Neisseria_gonorrhoeae': [251.82886017354, 25.182886017354, 377.082886017354],
               'Pseudomonas_aeruginosa': [390.797875891937, 39.0797875891937, 574.579787589194],
               'Salmonella_enterica': [900.357947376589, 90.0357947376589, 1120.53579473766],
               'Staphylococcus_aureus': [1252.9279205747, 125.29279205747, 1459.99279205747],
               'Streptococcus_pneumoniae': [2481.830587, 248.1830587, 3112.883059]}
    
    x = np.arange(1, var_freq.var_freq.max(), step = 1).tolist()
    bisect.insort(x, sol.x)
    bisect.insort(x, inf_pts[species][0])
    x = np.array(x)
    y = cfd(x, *params)
    y = rescale(pd.Series([0] + y.tolist()).diff()[1:].reset_index(drop=True), [var_freq.var_count.min(), var_freq.var_count.max()], back = True).cumsum()
    y_expected = cfd(x, *expected_fits[species])
    y_expected = rescale(pd.Series([0] + y_expected.tolist()).diff()[1:].reset_index(drop=True), [var_freq.var_count.min(), var_freq.var_count.max()], back = True).cumsum()
    plt.plot(var_freq.iloc[:,0], var_freq.iloc[:,4], label = 'Data', marker = 'o', color = 'magenta', alpha = 0.3, markersize = 4)
    plt.plot(x, y, label = 'Actual Fit', linestyle='dashdot', color = 'blue')
    plt.plot(x, y_expected, label = 'Expected Fit', linestyle='dashdot', color = 'orange')
    plt.plot(sol.x, y[x == sol.x], marker = 'o', label = 'Actual Inf. pt.', color = 'blue')
    plt.plot(inf_pts[species][0], y_expected[x == inf_pts[species][0]], marker = 'o', label = 'Expected Inf. pt.', color = 'orange')
    plt.axvline(sol.x, linestyle='dashed', color = 'grey')
    plt.axvline(inf_pts[species][0], linestyle='dashed', color = 'grey')
    plt.axvline(inf_pts[species][1], linestyle='dashed', label = 'Expected cutoffs', color = 'orange')
    plt.axvline(inf_pts[species][2], linestyle='dashed', color = 'orange')
    plt.axvline(unique_t, linestyle='dashed', label = 'Actual cutoffs', color = 'blue')
    plt.axvline(core_t, linestyle='dashed', color = 'blue')
    plt.xlabel('Gene Frequency in Genomes')
    plt.ylabel('Cumulative Gene Count')
    plt.legend()
    plt.title(species)
    plt.savefig(species+'/'+species+"_pangenome_division.png")

    if(save):
      var_counts['gene_class'] = np.repeat("", var_counts.shtotal_genomesape[0])
      var_counts.loc[core_filter, 'gene_class'] = 'core'
      var_counts.loc[accessory_filter, 'gene_class'] = 'accessory'
      var_counts.loc[unique_filter, 'gene_class'] = 'unique'

      var_counts.to_csv(species+'/'+species+'_pangenome.csv')
      var_freq.to_csv(species+'/'+species+'_distributions.csv')

start_time = time.time()

construct_pangenome(species, total_genomes)

print("\n======== Pangenome Construction Time for " + species + "========")
print("\t\t", (time.time() - start_time), " seconds\n")