#!/bin/bash

# Test the CNN on the Georgia dataset (https://www.kaggle.com/datasets/bjoernjostein/georgia-12lead-ecg-challenge-database)
NEWDIR="GeorgiaRefinementAll"
if [[ ! -d $NEWDIR ]]; then
	mkdir $NEWDIR
fi

RUN=10

for DIRECTORY in D1 D1-D2 12leads
do
	S=0

	while [ $S -lt $RUN ]
	do
		python GeorgiaRefinementAll.py --seed $S --path GeorgiaRefinementLastLayer/${DIRECTORY}/20Classes_$S --scenario $DIRECTORY --newpath ${NEWDIR}/${DIRECTORY}/20Classes_$S

		S=$((S+1))
	done
done

python CI_GeorgiaRefinementAll.py
