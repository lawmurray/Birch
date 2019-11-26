#!/bin/sh

echo "Bayesian linear regression model"
birch sample --config config/linear_regression.json

echo "Linear-Gaussian state-space model"
birch sample --config config/linear_gaussian.json

echo "Mixed linear-nonlinear-Gaussian state-space model"
birch sample --config config/mixed_gaussian.json

echo "SIR Markov model"
birch sample --config config/sir.json
