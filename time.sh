#!/bin/sh

birch sample \
  --seed 0 \
  --config config/time.json \
  --input input/filter.json \
  --output output/time${1}.json \
  --diagnostic diagnostic/time${1}.json
