ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# cuda11.1 + pytorch 1.9.0 + cudnn8 not work!!!
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime
ARG SERVER_MODE=prod

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8

# Install linux package
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
RUN	apt-get update && apt-get install -y gnupg2 git libglib2.0-0 \
    libgl1-mesa-glx curl wget zip ninja-build build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install ymir-exc sdk
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
# RUN if [ "${SERVER_MODE}" = "dev" ]; then \
#         pip install --force-reinstall -U "git+https://github.com/IndustryEssentials/ymir.git/@dev#egg=ymir-exc&subdirectory=docker_executor/sample_executor/ymir_exc"; \
#     else \
#         pip install ymir-exc; \
#     fi
ADD executor /executor
RUN pip install -e /executor

# install detectron2
# RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
# RUN git clone https://github.com/facebookresearch/detectron2.git && pip install -e detectron2

ADD . /workspace
RUN pip install -e . /workspace

WORKDIR /workspace
RUN echo "python3 start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh
