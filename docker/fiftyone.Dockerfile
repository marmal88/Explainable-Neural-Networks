# The base image to build from; must be Debian-based (eg Ubuntu)
ARG BASE_IMAGE=ubuntu:20.04
FROM $BASE_IMAGE

# The Python version to install
ARG PYTHON_VERSION=3.8

# Install system packages
RUN apt -y update \
    && apt -y --no-install-recommends install software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt -y update \
    && apt -y upgrade \
    && apt -y --no-install-recommends install tzdata \
    && TZ=Etc/UTC \
    && apt -y --no-install-recommends install \
        build-essential \
        ca-certificates \
        cmake \
        cmake-data \
        pkg-config \
        libcurl4 \
        libsm6 \
        libxext6 \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        unzip \
        curl \
        wget \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        ffmpeg \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -s /usr/local/lib/python${PYTHON_VERSION} /usr/local/lib/python \
    && curl https://bootstrap.pypa.io/get-pip.py | python \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip --no-cache-dir install --upgrade pip setuptools wheel ipython nbformat

# Install FiftyOne from source
RUN pip install fiftyone --no-binary fiftyone,voxel51-eta

# The name of the shared directory in the container that should be
# volume-mounted by users to persist data loaded into FiftyOne
EXPOSE 5151
COPY --chown=1000:1000 ./notebooks/ /notebooks/

CMD ipython --TerminalIPythonApp.file_to_run=/notebooks/fiftyone.ipynb
