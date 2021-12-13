FROM ubuntu:18.04
RUN apt update

COPY requirements.txt .
RUN apt-get install python3-pip -y 
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libjpeg-dev zlib1g-dev -y

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --no-cache-dir

RUN export BENTOML_HOME='/usr/src/bentoml_volume/bentoml/'
