#!/bin/bash

# Test the CNN on the China dataset (https://www.kaggle.com/datasets/bjoernjostein/china-12lead-ecg-challenge-database)
NEWDIR="ChinaRefinement"
if [[ ! -d $NEWDIR ]]; then
	mkdir $NEWDIR
fi

RUN=10

for DIRECTORY in D1 D1-D2 12leads
do
	S=0

	mkdir ${NEWDIR}/${DIRECTORY}

	while [ $S -lt $RUN ]
	do
		mkdir ${NEWDIR}/${DIRECTORY}/20Classes_$S

		python ChinaRefinement.py --seed $S --path GeorgiaRefinementAll/${DIRECTORY}/20Classes_$S --scenario $DIRECTORY --newpath ${NEWDIR}/${DIRECTORY}/20Classes_$S

		S=$((S+1))
	done
done

python CI_ChinaRefinement.py
