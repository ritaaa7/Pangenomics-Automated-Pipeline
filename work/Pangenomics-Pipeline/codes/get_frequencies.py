import pandas as pd 
import sys

filename = sys.argv[1]
csv_filename = sys.argv[2] + "/" + sys.argv[2] + '_cluster_frequencies.csv'

# Function that extracts the genome from a line in cluster
def get_genome(line):
    genomes_list = line.split("|")
    genome = genomes_list[1].split("...")[0]  # Extract only the ID
    return genome

f = open(filename, 'r')
lines = f.readlines()

cluster_count = 0
frequencies = {}

for line in lines:
	if line.startswith('>'):
		cluster_name = "Cluster " + str(cluster_count)
		frequencies[cluster_name] = 0
		genome_list = set()
		cluster_count += 1
	
	else:
		genome = get_genome(line)
		if genome in genome_list:
			continue 
		else: 
			genome_list.add(genome)
			frequencies[cluster_name] += 1

frequencies_table = pd.DataFrame(frequencies.items())
frequencies_table.columns = [" ", "Number of genomes"]

frequencies_table.to_csv(csv_filename)

