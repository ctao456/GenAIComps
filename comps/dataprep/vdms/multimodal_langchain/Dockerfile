# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ENV LANG=C.UTF-8

ARG ARCH="cpu"

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libcairo2-dev \
    libgl1-mesa-glx \
    libjemalloc-dev \
    vim

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

USER user

COPY comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    if [ ${ARCH} = "cpu" ]; then pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; fi && \
    pip install --no-cache-dir -r /home/user/comps/dataprep/vdms/multimodal_langchain/requirements.txt

ENV PYTHONPATH=/home/user

USER root

RUN mkdir -p /home/user/comps/dataprep/vdms/multimodal_langchain/uploaded_files && chown -R user /home/user/comps/dataprep/vdms/multimodal_langchain

USER user

WORKDIR /home/user/comps/dataprep/vdms/multimodal_langchain

ENTRYPOINT ["python", "ingest_videos.py"]
