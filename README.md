# Explainable-Neural-Networks





# Introduction

Following is the initial file structure. 
# Folders/File structure 
```bash
.
├── data
│   ├── test
│   │   ├── NORMAL
│   │   └── PNEUMONIA
│   ├── train
│   │   ├── NORMAL
│   │   └── PNEUMONIA
│   └── val
│       ├── NORMAL
│       └── PNEUMONIA
├── conda-env.yaml
├── src
├── LICENSE
├── notebooks
└── README.md
```

# Getting Started

1. Data

Curl down and unzip the data to the existing repositry using the command:
```bash
# download using
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BLViYnohD-S4u5p1DkXp1MOlCGe02w36' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BLViYnohD-S4u5p1DkXp1MOlCGe02w36" -O pneumonia.zip && rm -rf /tmp/cookies.txt
# Unzip using
unzip -q pneumonia.zip -d data/
```

2. Environment

Please ensure dependencies adhere to python 3.10
```bash
conda create -n xnn python=3.10
conda activate xnn
```
1. Pre-commit hook

More information on precommit hook [here](https://pre-commit.com/).
```bash
pre-commit install
```

# Contributing
- Outstanding features are listed in the project's kanban board [here](https://github.com/users/marmal88/projects/4/views/2)!
- Dont see an issue you want? raise an issue [here](https://github.com/marmal88/Explainable-Neural-Networks/issues)
