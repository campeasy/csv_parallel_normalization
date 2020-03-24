#    AY 19/20
#    Salvatore Campisi
#    Parallel Programming on GPU
#    CSV Parallel Normalization

#   run.sh
#   Script for launching the project

if [ $# -eq 2 ]
then
    # Launching the host program:
    cd bin
    export OCL_PLATFORM=$1 && export OCL_DEVICE=$2 && ./main

    # Going back to previous folder:
    cd ..
else
    echo "zsh run.sh OCL_PLATFORM_VALUE OCL_DEVICE_VALUE"
fi
