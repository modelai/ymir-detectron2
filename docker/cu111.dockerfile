ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# cuda11.1 + pytorch 1.9.0 + cudnn8 not work!!!
# youdaoyzbx/ymir-executor:detectron2-tmi
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime
ARG SERVER_MODE=prod
ARG YMIR="1.0.0" # 1.0.0, 1.1.0 or 1.2.0

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8
ENV YMIR_VERSION=${YMIR}
ENV PYTHONPATH=.

# Install linux package
RUN	apt-get update && apt-get install -y gnupg2 git libglib2.0-0 \
    libgl1-mesa-glx vim curl wget zip ninja-build build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install ymir-exc sdk
RUN if [ "${SERVER_MODE}" = "dev" ]; then \
        pip install "git+https://github.com/IndustryEssentials/ymir.git/@dev#egg=ymir-exc&subdirectory=docker_executor/sample_executor/ymir_exc"; \
    else \
        pip install ymir-exc; \
    fi

COPY . /workspace
RUN pip install -r requirements.txt && pip install -e . /workspace \
    && mkdir -p /img-man && mv /workspace/ymir/img-man/* /img-man

WORKDIR /workspace
RUN echo "python3 start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh
