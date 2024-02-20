#!/usr/bin/env bash

mkdir data
mkdir data/models
mkdir data/paths
wget https://nlp.stanford.edu/data/glove.6B.zip
wget https://zenodo.org/records/3876154/files/causenet-precision.jsonl.bz2
wget https://zenodo.org/records/3876154/files/causenet-sample.json
mv glove.6B.zip causenet-precision.jsonl.bz2 causenet-sample.json data/

wget https://www.dropbox.com/scl/fi/y1cn9khbtjj0x5eme237p/models.tar.gz?rlkey=dh35zjtq8g7rskz05e42uopk2&e=1&dl=0
tar xf models.tar.gz?rlkey=dh35zjtq8g7rskz05e42uopk2

cd src
pip install .
python -m nltk.downloader stopwords punkt wordnet omw-1.4
