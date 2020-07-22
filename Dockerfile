# Dockerfile with tensorflow gpu support on python3, opencv3.3
FROM tensorflow/tensorflow:1.15.2-py3-jupyter

MAINTAINER Inho C. <hasbegun@gmail.com>

# The code below is all based off the repos made by https://github.com/janza/
# He makes great dockerfiles for opencv, I just used a different base as I need
# tensorflow on a gpu.

# Core linux dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libhdf5-dev \
    libpq-dev \
    vim sudo

# Python dependencies
RUN pip3 --no-cache-dir install \
    numpy \
    hdf5storage \
    h5py \
    scipy \
    py3nvml

ENV OPENCV_VERSION="3.4.6"
RUN cd /opt && \
    wget --quiet https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    rm -rf ${OPENCV_VERSION}.zip && \
    wget --quiet https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    rm -rf ${OPENCV_VERSION}.zip && \
    mkdir opencv-${OPENCV_VERSION}/build && \
    cd opencv-${OPENCV_VERSION}/build && \
    cmake -DBUILD_TIFF=ON \
        -DBUILD_opencv_java=OFF \
        -DWITH_CUDA=OFF \
        -DENABLE_AVX=ON \
        -DWITH_OPENGL=ON \
        -DWITH_OPENCL=ON \
        -DWITH_IPP=ON \
        -DWITH_TBB=ON \
        -DWITH_EIGEN=ON \
        -DWITH_V4L=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
        .. && \
    make -j4 && make -j4 install

# Clean up
RUN rm -rf /opt/opencv-${OPENCV_VERSION} /opt/opencv_contrib-${OPENCV_VERSION}

RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev


WORKDIR /project
ADD requirements-docker.txt requirements.txt
RUN pip3 install -r requirements.txt
ADD . .
RUN python setup.py install
RUN pip3 install keras==2.3.1

# install coco
RUN cd /opt && \
    git clone https://github.com/waleedka/coco.git && \
    cd coco/PythonAPI && \
    python3 setup.py install && \
    cd / && rm -rf /opt/coco

# redis will be taken care of by docker compose.
RUN apt-get update && \
    apt-get install -y \
    redis-server

## display python gui outputs
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -qq \
    python3-tk

RUN rm -rf /project

# add user
RUN useradd -m developer && \
    usermod -aG sudo developer && \
    echo '%sudo ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers && \
    cp /root/.bashrc /home/developer/ && \
    mkdir /home/developer/projects && \
    chown -R --from=root developer:developer /home/developer

# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV HOME /home/developer
ENV USER developer
USER developer
ENV PATH /home/developer/.local/bin:$PATH
# Avoid first use of sudo warning. c.f. https://askubuntu.com/a/22614/781671
RUN touch $HOME/.sudo_as_admin_successful

WORKDIR /home/developer/projects

#ENTRYPOINT ["/project/entrypoint.sh"]
#CMD ["run"]
CMD ["/bin/bash"]