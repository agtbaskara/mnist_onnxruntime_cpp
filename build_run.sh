#!/bin/bash

mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..
./build/app
