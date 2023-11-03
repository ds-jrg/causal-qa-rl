#!/usr/bin/env bash

mkdir data
mkdir data/models
wget https://nlp.stanford.edu/data/glove.6B.zip
wget https://groups.uni-paderborn.de/wdqa/causenet/causality-graphs/causenet-precision.jsonl.bz2
wget https://groups.uni-paderborn.de/wdqa/causenet/causality-graphs/causenet-sample.json
mv glove.6B.zip causenet-precision.jsonl.bz2 causenet-sample.json data/

wget https://www.dropbox.com/s/gdd3qllyzs0mw32/models.tar.gz
tar xf models.tar.gz

cd src
python setup.py develop
