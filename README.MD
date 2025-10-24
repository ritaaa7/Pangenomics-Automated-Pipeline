
from terminal 

1- clone the repo and change the dirctory
2-docker compose build 
3-docker compose up -d
4- get the container id: docker ps
5-connect to the container: docker exec -it {container id } bash
6- cd codes
7- run the workflow: ./workflow1.sh



CD HIT 
cd-hit -i /work/input.faa -o /work/out.faa -c 0.95 -n 5

EGG NOG
emapper.py -i /work/input.faa -o /work/test_v7 --itype proteins -m diamond --dmnd_db /data/eggnog/eggnog_proteins.dmnd --no_annot --cpu 4

need to manually add the eggnog databases because they're too large to add in the repo


to run the workflow1.sh go into codes directory + we need to provide the path of genome_ids_fasta: /work/Pangenomics-Pipeline/genome_ids_fasta

for the fischer test to work: 
-we changed the adjust eggnoug output code.
-We also changed the cut command in workflow1.sh and commented out the df = df.drop[0] in all fisher test.py files
