#!/bin/zsh

# Cleaning and compiling the binary file:
echo "--------------------------------------------------"
echo "      Parallel Normalization - Program Build      "
echo "--------------------------------------------------"
make clean
echo "\n[OK] Binary files directory cleaned\n"
make make
echo "\n[OK] Host program correctly compiled\n"

# Launching the host program:
cd bin
export OCL_PLATFORM=0 && export OCL_DEVICE=1 && ./main

# Going back to previous folder:
cd ..
