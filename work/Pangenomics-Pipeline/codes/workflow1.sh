#!/bin/bash

# Retrieve path to directory with csv file(s) containing the genome IDs from user 
read -p "Enter the path to the directory containing the genome IDs: " directory
echo

# Ask user for the total number of genomes
read -p "Enter the total number of genomes: " total_genomes
echo

# Keeps track of the run-time of the whole workflow 
start_time=$(date +%s) 

# Loop through each file in the provided directory
for file in "$directory"/*.csv; do

	# Keep the name of the species in a variable called species_name 
	# Example : Enterobacter_cloacae
	species_name=${file#$directory/}
	species_name=${species_name%'_genome_ids.csv'}

	# Create a directory for each species processed and store all files related to it inside it
	mkdir $species_name

	# Check if CSV file provided has the appropriate header 
	desired_line="genome.genome_ids"
	# Extract the first line of the CSV file
	first_line=$(head -n 1 $file)
	# Compare the first line with the desired line
	if [ "$first_line" != "$desired_line" ]; then
	    echo $desired_line | cat - $file > temp && mv temp $file
	    rm temp
	fi
	
	# Store all ids to loop through them 
    genome_ids=$(tail -n +2 $file)

    # Store the name that will be used for the CDS fasta file in the variable cds_name
    cds_fname=${species_name}"_CDS"

    echo "______________________________________________________"
    echo
    echo "Extracting CDS from Genome IDs of $species_name"
    echo "______________________________________________________"
    echo

    # Keeps track of CDS extraction run-time from all the genome IDs of each species 
    cds_time=$(date +%s)
    
    # Loop through each id and concatenate the CDS sequences to one output file for each species 
    for id in $genome_ids; do
    	if [ -e "$species_name/$cds_fname.fasta" ]; then
    		echo "Extracting CDS from genome ID $id"
    		p3-genome-fasta --protein $id >> $species_name/$cds_fname.fasta 
    	else
    		echo "Creating new CDS fasta file for $species_name"
    		echo "Extracting CDS from genome ID $id"
    		p3-genome-fasta --protein $id > $species_name/$cds_fname.fasta
    	fi
    done

    run_time=$(($(date +%s) - cds_time)) 
	echo
	echo "----- Total Time for CDS Extraction: $run_time seconds-----"
	echo

    # ---------------------Genome Clustering---------------------

	echo "______________________________________________________"
	echo
	echo "Running Genome Clustering Step for $species_name"
	echo "______________________________________________________"
	echo

	# Keeps track of the genome clustering run-time 
	clustering_time=$(date +%s)

	# CD-HIT command 
	cd-hit -i $species_name/$cds_fname.fasta -o $species_name/$species_name.fasta -n 5 -c 0.8 -aL 0.8 -M 16000 -T 16
	
	run_time=$(($(date +%s) - clustering_time)) 
	echo
	echo "----- Total Time for Genome Clustering: $run_time seconds-----"
	echo

	# Counts the number of unique genomes in each cluster for later use 
	python3 get_frequencies.py $species_name/$species_name.fasta.clstr $species_name

	# ---------------------Frequency-based Division of Pangenome---------------------

	echo "______________________________________________________"
	echo
	echo "Running Pangenome Construction Step for $species_name"
	echo "______________________________________________________"
	echo


	

	python3 pangenome_construction.py $species_name $total_genomes

        
	# Translate CDS (DNA) into protein FASTA				# 
	cds_input="$species_name/$cds_fname.fasta"				#
	protein_output="$species_name/${species_name}_proteins.faa"		#
										#
	echo "Translating CDS file into protein sequences..."			#
	python3 translate_cds_to_proteins.py "$cds_input" "$protein_output"	#



	# ---------------------Functional Annotation eggNOG-mapper ---------------------

	# Keeps track of eggNOG-mapper run-time 
	eggNOG_time=$(date +%s) 

	echo "______________________________________________________"
	echo
	echo "Running eggNOG-mapper Step for $species_name"
	echo "______________________________________________________"
	echo

	emapper.py -i "$protein_output" -o $species_name/$species_name --usemem --cpu 16 --outfmt_short     # $species_name/$species_name.fasta --> $protein_output

	run_time=$(($(date +%s) - eggNOG_time)) 
	echo
	echo "----- Total Run Time for eggNOG-mapper: $run_time seconds-----"
	echo
	
	# Extract only COG Category and GO terms and adjust file format 
	tail -n +5 $species_name/$species_name.emapper.annotations | cut -f 1,7,10 > $species_name/${species_name}_COG_GO.csv
	python3 adjust_eggNOG_output.py $species_name/${species_name}_COG_GO.csv

	echo "================================================="
	echo
	echo "Building Contingency Table for $species_name"
	echo

	numb_genes=$(grep -c "^>" $species_name/$species_name.fasta)
	python3 perform_fisher_test_core.py $species_name/${species_name}_COG_GO.csv $species_name/$species_name.fasta $species_name/${species_name}_pangenome.csv $species_name $numb_genes

done

end_time=$(($(date +%s) - start_time)) 

echo
echo "----- Total Workflow Run-time: $run_time seconds-----"
echo

echo "Generating Visualizations"
python visualizations3.py
