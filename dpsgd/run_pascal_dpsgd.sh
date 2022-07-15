#!/bin/bash

DPMETHODS=(True)
BATCHSIZES=(32)
CCLIPS=(5)
clear

for CCLIP in "${CCLIPS[@]}"
do	
for DPMETHOD in "${DPMETHODS[@]}"
do
	for BATCHSIZE in "${BATCHSIZES[@]}"
	do
		echo "DP $DPMETHOD and batch_size=$BATCHSIZE and Clip=$CCLIP"
		python Pascal_DPSGD.py --EPOCHS=50 --DPSGD=$DPMETHOD --BATCH_SIZE=$BATCHSIZE --NOISE_MULTIPLIER=0.5 --CCLIP=$CCLIP
	done
done
done
