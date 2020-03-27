#    AY 19/20
#    Salvatore Campisi
#    Parallel Programming on GPU
#    CSV Parallel Normalization

#   run.sh
#   Script for launching the project

if [ $# -ge 4 ]
then
	 # Data Backup:
    rm data/normalized.csv
	 cp $3 data/backup.csv

    # Launching the host program:
    cd bin
    export OCL_PLATFORM=$1 && export OCL_DEVICE=$2 && ./main "${@:3}"
    cd ..

	 # Setting the correct names for new data:
	 mv $3 data/normalized.csv
	 mv data/backup.csv $3

else
    echo "[FAIL] Example of use: zsh run.sh OCL_PLATFORM_VALUE OCL_DEVICE_VALUE csv_pathname_to_normalize col_index1 col_index2 ... col_indexN"
	 echo "                       zsh run.sh OCL_PLATFORM_VALUE OCL_DEVICE_VALUE csv_pathname_to_normalize ALL"
fi
