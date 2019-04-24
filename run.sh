#!/bin/sh

birch sample --config config/linear_regression.json --input input/bike_share.json --output output/linear_regression.json
birch sample --config config/linear_gaussian.json --input input/linear_gaussian.json --output output/linear_gaussian.json
birch sample --config config/mixed_gaussian.json --input input/mixed_gaussian.json --output output/mixed_gaussian.json
birch sample --config config/sir.json --input input/russian_influenza.json --output output/sir.json
