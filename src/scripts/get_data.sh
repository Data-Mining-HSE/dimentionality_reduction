#!bin/bash

mkdir -p data
pushd data
wget https://github.com/Data-Mining-HSE/data_directory/archive/refs/heads/main.zip .
unzip main.zip
unzip data_directory-main/dimentionality_reduction.zip
mv __MACOSX/dimentionality_reduction dimentionality_reduction 
ls | grep -v dimentionality_reduction | xargs rm -rfv
popd