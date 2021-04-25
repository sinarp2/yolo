# Build image
FROM ubuntu:18.04 AS build
RUN sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y build-essential git

# Get and compile darknet
WORKDIR /src
#RUN git clone -n https://github.com/AlexeyAB/darknet.git
#WORKDIR /src/darknet
#RUN git checkout 4c315ea26b56c2bf20ebc240d94386c6e3cc83db
RUN git clone https://github.com/AlexeyAB/darknet.git
WORKDIR /src/darknet
# RUN sed -i -e "s!OPENMP=0!OPENMP=1!g" Makefile && \
#     sed -i -e "s!OPENCV=0!OPENCV=1!g" Makefile && \
#     sed -i -e "s!AVX=0!AVX=1!g" Makefile && \
#     sed -i -e "s!LIBSO=0!LIBSO=1!g" Makefile && \
#     make
RUN sed -i -e "s!OPENMP=0!OPENMP=1!g" Makefile && \
    sed -i -e "s!AVX=0!AVX=1!g" Makefile && \
    sed -i -e "s!LIBSO=0!LIBSO=1!g" Makefile && \
    make

# App image:
FROM ubuntu:18.04

# Bare-bones python install
RUN sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y libgomp1 wget && \
    apt-get install -y --no-install-recommends python3-pip && \
    apt-get install -y python3-setuptools && \
    pip3 install --upgrade pip && \
    pip3 install --upgrade flask && \
    pip3 install --no-cache-dir wheel opencv-python-headless && \
    rm -rf /var/lib/apt/lists

# Get darknet from build image
WORKDIR /app
COPY --from=build /src/darknet/libdarknet.so .
COPY --from=build /src/darknet/darknet.py .
COPY --from=build /src/darknet/cfg cfg/
COPY --from=build /src/darknet/data data/
ENV DARKNET_PATH=/app PYTHONPATH=/app