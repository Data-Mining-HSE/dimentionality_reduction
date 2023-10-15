#!bin/bash

mkdir -p temp


pushd temp
wget https://github.com/Data-Mining-HSE/data_directory/archive/refs/heads/main.zip .
unzip main.zip
unzip data_directory-main/dimentionality_reduction.zip
mv dimentionality_reduction ../data
popd
rm -rf temp
