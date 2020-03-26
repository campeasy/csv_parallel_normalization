#    AY 19/20
#    Salvatore Campisi
#    Parallel Programming on GPU
#    CSV Parallel Normalization

#   run.sh
#   Script for launching the project

if [ $# -eq 2 ]
then
	 # Data Backup:
	 cd data
    rm normalized.csv
	 cp credit_card_fraud_PCA.csv backup.csv
	 cd ..

    # Launching the host program:
    cd bin
    export OCL_PLATFORM=$1 && export OCL_DEVICE=$2 && ./main
    cd ..

	 # Setting the correct names:
	 cd data
	 mv credit_card_fraud_PCA.csv normalized.csv
	 mv backup.csv credit_card_fraud_PCA.csv
	 cd ..

else
    echo "zsh run.sh OCL_PLATFORM_VALUE OCL_DEVICE_VALUE"
fi
