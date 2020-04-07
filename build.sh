#    AY 19/20
#    Salvatore Campisi
#    Parallel Programming on GPU
#    CSV Parallel Normalization

#   build.sh
#   Script for building the project

# Cleaning and compiling the binary file:
echo "--------------------------------------------------"
echo "      Parallel Normalization - Program Build      "
echo "--------------------------------------------------"
make clean
rm -rf bin/tests && rm -rf bin/
mkdir bin && mkdir bin/tests
echo "\n[OK] Binary files directory cleaned\n"
make make
echo "\n[OK] Host program correctly compiled\n"
