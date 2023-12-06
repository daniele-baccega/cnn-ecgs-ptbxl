#!/bin/bash

# Test the CNN on the Georgia dataset (https://www.kaggle.com/datasets/bjoernjostein/georgia-12lead-ecg-challenge-database)
# Read this to correctly download the Georgia dataset using kaggle command: https://medium.com/@c.venkataramanan1/setting-up-kaggle-api-in-linux-b05332cde53a
if [[ ! -d Georgia ]]; then
	kaggle datasets download -d bjoernjostein/georgia-12lead-ecg-challenge-database
	unzip -qq georgia-12lead-ecg-challenge-database.zip
	rm georgia-12lead-ecg-challenge-database.zip
	mv WFDB Georgia
fi

for DIRECTORY in D1 D1-D2 D1-V1 D1-V2 D1-V3 D1-V4 D1-V5 D1-V6 8leads 12leads 12leads_WithoutDataAugmentation
do
	for i in {0..49}
	do
		python Georgia_Test.py --path TrainedModels/${DIRECTORY}/20Classes_$i --scenario $DIRECTORY
	done
done

python CI_Georgia.py
