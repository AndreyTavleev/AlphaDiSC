FROM ghcr.io/hombit/mesa2py:latest

COPY README.md setup.py pyproject.toml LICENSE.md requirements.txt /app/
COPY tests /app/tests/

RUN pip3 install -U pip setuptools

COPY alpha_disc /app/alpha_disc/

WORKDIR /app

RUN pip3 install .
