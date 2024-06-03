
FROM python:3.10-slim

# COPY ./requirements.txt /tmp/requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install altair
RUN pip install pandas
RUN pip install streamlit