#!/bin/bash

sudo apt update
sudo apt upgrade
# Install pip
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
sudo mkdir /python && cd /python
sudo wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0a7.tgz
sudo apt install software-properties-common
sudo apt install python3.10
sudo apt install python3-pip
# Install mlflow
pip install --upgrade setuptools
pip install mlflow
# Check version of mlflow installed
echo mlflow --version