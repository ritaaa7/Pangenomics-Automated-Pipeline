
#### From terminal 

1. git clone https://github.com/ritaaa7/Pangenomics.git
2. cd Pangenomics
3. sudo docker compose build 
4. docker compose up -d
5. docker ps
6. docker exec -it {container id} bash
7. cd codes
8. ./workflow1.sh when prompted to enter the directory containing the genome IDs: 
/work/Pangenomics-Pipeline/genome_ids_fasta

need to manually add the eggnog databases because they're too large to add in the repo

#### for the fischer test to work: 
- we changed the adjust eggnoug output code.
- We also changed the cut command in workflow1.sh and commented out the df = df.drop[0] in all fisher test.py files
