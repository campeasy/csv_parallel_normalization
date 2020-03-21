#!/bin/zsh

# Cleaning and compiling the binary file:
make clean
make make

# Setting up the environment variables:
export OCL_PLATFORM=0
export OCL_DEVICE=1

clear

# Running the main binary file:
./bin/main
