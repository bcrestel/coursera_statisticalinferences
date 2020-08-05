FROM python:3.7

ADD requirements.txt ./
RUN pip install -r requirements.txt

RUN jupyter contrib nbextension install --user && \
    jupyter nbextension enable toc2/main && \
    jupyter nbextension enable execute_time/ExecuteTime && \
    jupyter nbextension enable collapsible_headings/main && \
    jupyter nbextension enable move_selected_cells/main && \
    jupyter nbextensions_configurator enable --user
