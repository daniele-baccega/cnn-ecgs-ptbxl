#!/bin/bash

# Test the CNN on the China dataset (https://www.kaggle.com/datasets/bjoernjostein/china-12lead-ecg-challenge-database)
NEWDIR="ChinaRefinementAll"
if [[ ! -d $NEWDIR ]]; then
	mkdir $NEWDIR
fi

RUN=10

for DIRECTORY in D1 D1-D2 12leads
do
	S=0

	while [ $S -lt $RUN ]
	do
		python ChinaRefinementAll.py --seed $S --path ChinaRefinementLastLayer/${DIRECTORY}/20Classes_$S --scenario $DIRECTORY --newpath ${NEWDIR}/${DIRECTORY}/20Classes_$S

		S=$((S+1))
	done
done

python CI_ChinaRefinementAll.py
