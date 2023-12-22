#!/bin/bash

# Test the CNN on the China dataset (https://www.kaggle.com/datasets/bjoernjostein/china-12lead-ecg-challenge-database)
NEWDIR="ChinaRefinementLastLayer"
if [[ ! -d $NEWDIR ]]; then
	mkdir $NEWDIR
fi

RUN=10

for DIRECTORY in D1 D1-D2 12leads
do
	S=0

	while [ $S -lt $RUN ]
	do
		python ChinaRefinementLastLayer.py --seed $S --path GeorgiaRefinementAll/${DIRECTORY}/20Classes_$S --scenario $DIRECTORY --newpath ${NEWDIR}/${DIRECTORY}/20Classes_$S

		S=$((S+1))
	done
done

python CI_ChinaRefinementLastLayer.py
