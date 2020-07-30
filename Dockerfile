FROM python:3.7

ADD requirements.txt ./
RUN pip install -r requirements.txt

RUN jupyter nbextensions_configurator enable --user && \
    jupyter contrib nbextension install --user
