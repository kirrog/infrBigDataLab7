FROM spark:3.5.1-scala2.12-java17-python3-ubuntu

WORKDIR /app

USER root

ENV PYTHONPATH="${PYTHONPATH}:/app"

COPY . /app

RUN ln -sf $(which python3) /usr/bin/python && \
    ln -sf $(which pip3) /usr/bin/pip

RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
