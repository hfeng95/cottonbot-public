FROM continuumio/miniconda3

WORKDIR /
RUN conda env create -f /environment.yml
