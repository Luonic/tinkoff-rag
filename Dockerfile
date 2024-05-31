FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt