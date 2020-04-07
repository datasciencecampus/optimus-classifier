#!/usr/bin/env zsh

# install dependencies
pip3 install -r requirements.txt


#clone fasttext repo and install
git clone https://github.com/facebookresearch/fasttext
cd fasttext
pip3 install .
cd ../ && rm -rf fastext

# download binary FT  pretrained model and unzip
cd model
wget https://dl.fbaipublicfiles/fasttext/vectors-wiki/wiki.en.zip
unzip wiki.en.zip
cd ..
