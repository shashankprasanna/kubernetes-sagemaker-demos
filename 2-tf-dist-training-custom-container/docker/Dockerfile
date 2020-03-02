FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.15.2-gpu-py27-cu100-ubuntu18.04

RUN pip install sagemaker-containers

COPY cifar10-multi-gpu-horovod-sagemaker.py /opt/ml/code/cifar10-multi-gpu-horovod-sagemaker.py
COPY model_def.py /opt/ml/code/model_def.py

ENV SAGEMAKER_PROGRAM cifar10-multi-gpu-horovod-sagemaker.py
