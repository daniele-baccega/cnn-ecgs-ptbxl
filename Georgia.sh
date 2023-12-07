#!/bin/bash

# Test the CNN on the Georgia dataset (https://www.kaggle.com/datasets/bjoernjostein/georgia-12lead-ecg-challenge-database)
# Read this to correctly download the Georgia dataset using kaggle command: https://medium.com/@c.venkataramanan1/setting-up-kaggle-api-in-linux-b05332cde53a
if [[ ! -d Georgia ]]; then
	kaggle datasets download -d bjoernjostein/georgia-12lead-ecg-challenge-database
	unzip -qq georgia-12lead-ecg-challenge-database.zip
	rm georgia-12lead-ecg-challenge-database.zip
	mv WFDB Georgia
fi

NEWDIR="GeorgiaRefinement"
if [[ ! -d $NEWDIR ]]; then
	mkdir $NEWDIR
fi

RUN=10

for DIRECTORY in D1 D1-D2 D1-V1 D1-V2 D1-V3 D1-V4 D1-V5 D1-V6 8leads 12leads 12leads_WithoutDataAugmentation
do
	S=0

	while [ $S -lt $RUN ]
	do
		python Georgia.py --seed $S --path TrainedModels/${DIRECTORY}/20Classes_$S --scenario $DIRECTORY --newpath ${NEWDIR}/${DIRECTORY}/20Classes_$S &

		S=$((S+1))

		if [[ $S -lt $RUN ]]; then
			python Georgia.py --seed $S --path TrainedModels/${DIRECTORY}/20Classes_$S --scenario $DIRECTORY --newpath ${NEWDIR}/${DIRECTORY}/20Classes_$S

			S=$((S+1))
		fi
	done
done

python CI_Georgia.py
