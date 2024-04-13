FROM python:3.9.7-slim

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx
RUN apt-get install -y git

RUN pip install --upgrade pip
RUN pip3 install pika
RUN pip3 install ujson
# RUN pip3 install torch>=1.4.0
# RUN pip3 install torchvision
# RUN pip3 install pillow
# RUN pip3 install numpy
# RUN pip3 install tqdm
# RUN pip3 install timm
# RUN pip3 install easydict
# RUN pip3 install pyyaml==5.4.1
RUN pip3 install opencv-python
# RUN pip3 install asyncio
# RUN pip3 install aio-pika
RUN pip3 install onnx
RUN pip3 install onnxruntime
RUN pip3 install scipy
RUN pip3 install py-feat
RUN pip3 install python-dotenv

RUN pip3 install git+https://github.com/Trabajo-profesional-grupo-21/common.git@0.0.2#egg=common

COPY / /

VOLUME /src

CMD ["python3", "./main.py"]