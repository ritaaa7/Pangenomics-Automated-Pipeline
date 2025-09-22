#!/bin/bash
set -e

echo "Bioinformatics container started"
echo "Working directory: /work"
echo "EggNOG DB directory: /data/eggnog"
echo ""

if [ $# -eq 0 ]; then
    echo "Usage:"
    echo "  cd-hit -i /work/input.faa -o /work/out.faa -c 0.95 -n 5"
    echo "  emapper.py -i /work/input.faa -o /work/out --data_dir /data/eggnog --itype proteins --cpu 8 --dmnd"
    echo ""
    echo "Drop FASTA files into ./work on the host and they’ll appear in /work inside the container."
    exec bash   # open interactive shell if no args
else
    echo "Running command: $@"
    exec "$@"
fi