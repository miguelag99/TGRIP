#!/bin/bash

# Check if CARLA directory already exists
if [ -d "CARLA" ]; then
    echo "CARLA directory already exists. Skipping creation."
else
    # Download and install CARLA
    echo "Creating CARLA directory and downloading CARLA..."
    mkdir CARLA
    cd CARLA
    wget --content-disposition https://tiny.carla.org/carla-0-9-16-linux
    tar -xf CARLA_0.9.16.tar.gz
    rm CARLA_0.9.16.tar.gz
    cd Import
    wget --content-disposition https://tiny.carla.org/additional-maps-0-9-16-linux
    cd ..
    ./ImportAssets.sh
    cd ..
fi