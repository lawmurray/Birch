#!/bin/sh

# simulate a new data set
birch sample --config config/simulate.json --input input/simulate.json --output output/simulate.json
birch draw --input output/simulate.json --output figs/simulate.pdf
birch data --input output/simulate.json --output input/filter.json

# run the particle filter on the data set
birch sample --config config/filter.json --input input/filter.json --output output/filter.json
birch draw --input output/filter.json --output figs/filter.pdf
