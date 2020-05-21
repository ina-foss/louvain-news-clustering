FROM continuumio/miniconda3

#ENV http_proxy=http://firewall.ina.fr:81/
#ENV https_proxy=http://firewall.ina.fr:81/

#RUN git clone https://github.com/bmaz/twembeddings.git
RUN python -m pip install --upgrade pip
COPY twembeddings twembeddings
RUN pip install ./twembeddings/
RUN apt-get -y update && apt-get -y install build-essential libxml2-dev zlib1g-dev python-dev python-pip pkg-config libffi-dev libcairo-dev
RUN pip install python-igraph
RUN pip install louvain

ARG UID
ARG USERNAME
RUN groupadd -r -g ${UID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${UID} -o -s /bin/bash ${USERNAME}
USER ${USERNAME}
