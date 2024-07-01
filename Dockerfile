FROM python:3.11-slim

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx
RUN apt-get install -y git

RUN pip install --upgrade pip
RUN pip install scipy==1.11.0
RUN pip install py-feat
RUN pip install pika
RUN pip install ujson
RUN pip install pillow
RUN pip install opencv-python
RUN pip install onnx
RUN pip install onnxruntime
RUN pip install pydantic-settings
RUN pip install pydantic

RUN pip install git+https://github.com/Trabajo-profesional-grupo-21/common.git@1.0.0#egg=common

COPY / /

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

VOLUME /src


CMD ["python3", "./main.py"]