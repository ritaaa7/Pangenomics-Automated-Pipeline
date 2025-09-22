FROM mambaorg/micromamba:1.5.8

ENV MAMBA_DOCKERFILE_ACTIVATE=1


RUN micromamba install -y -n base -c conda-forge -c bioconda \
    python=3.10 \
    cd-hit=4.8.1 \
    eggnog-mapper=2.1.12 \
    diamond=2.1.9 \
    hmmer=3.4 \
    mmseqs2=15.6f452 \
 && micromamba clean -a -y

ENV EGGNOG_DATA_DIR=/data/eggnog


WORKDIR /work
ENV PATH="/opt/conda/bin:$PATH"

USER root 
RUN mkdir -p ${EGGNOG_DATA_DIR} /work
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh





ENTRYPOINT ["/usr/local/bin/start.sh"]
