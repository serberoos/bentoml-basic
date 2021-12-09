FROM ubuntu:18.04
RUN apt update

COPY requirements.txt .
RUN apt-get install python3-pip -y 
RUN pip3 install -r requirements.txt --no-cache-dir
RUN export BENTOML_HOME='/usr/src/bentoml_volume/bentoml/'
