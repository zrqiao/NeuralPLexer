FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        make \
        git \
        wget \
        tzdata \
        awscli \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

SHELL ["/bin/bash", "--login", "-c"]

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda init bash

RUN conda create -n NeuralPLexer python=3.9

RUN echo "conda activate NeuralPLexer" >> ~/.bashrc

RUN conda run -n NeuralPLexer pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir
RUN conda run -n NeuralPLexer pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html --no-cache-dir
RUN conda run -n NeuralPLexer pip install "git+https://github.com/facebookresearch/pytorch3d.git" --force-reinstall --no-deps --no-cache-dir

RUN git clone https://github.com/aqlaboratory/openfold.git
RUN cd openfold && sed -i 's/std=c++14/std=c++17/g' setup.py && conda run -n NeuralPLexer pip install . && cd ../ && rm -rf openfold

COPY NeuralPLexer.yml .
COPY NeuralPLexer-requirements.txt .
RUN conda run -n NeuralPLexer conda env update --file NeuralPLexer.yml && conda run -n NeuralPLexer conda clean -afy

RUN git clone https://github.com/zrqiao/NeuralPLexer.git && cd NeuralPLexer && conda run -n NeuralPLexer make install

ENTRYPOINT ["conda", "run", "-n", "NeuralPLexer"] 
