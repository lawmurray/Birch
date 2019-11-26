#!/bin/sh

echo "Simulating"
birch sample --config config/simulate.json --input input/simulate.json --output output/simulate.json

echo "Visualizing simulation"
birch draw --input output/simulate.json --output figs/simulate.pdf

echo "Creating data set from simulation"
birch data --input output/simulate.json --output input/filter.json

echo "Running particle filter"
birch sample --config config/filter.json --input input/filter.json --output output/filter.json

echo "Visualizing filter"
birch draw --input output/filter.json --output figs/filter.pdf
