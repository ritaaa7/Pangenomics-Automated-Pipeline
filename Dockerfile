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

USER root

# Install Perl dependencies and BV-BRC CLI
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    perl \
    libtext-table-perl \
    libtime-hires-perl \
    libtry-tiny-perl \
    liburi-perl \
    libxml-libxml-perl \
    libyaml-perl \
 && wget https://github.com/BV-BRC/BV-BRC-CLI/releases/download/1.040/bvbrc-cli-1.040.deb \
 && dpkg -i bvbrc-cli-1.040.deb || apt-get -f install -y \
 && rm bvbrc-cli-1.040.deb \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work
ENV PATH="/opt/conda/bin:$PATH"

USER root 
RUN mkdir -p ${EGGNOG_DATA_DIR} /work
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh





ENTRYPOINT ["/usr/local/bin/start.sh"]