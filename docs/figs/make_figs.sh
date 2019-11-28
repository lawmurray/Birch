#!/bin/sh

dot -Tsvg:cairo MarkovModel.dot > MarkovModel.svg
dot -Tsvg:cairo HiddenMarkovModel.dot > HiddenMarkovModel.svg

# svg:cairo does not seem to support links, so use just svg for these
dot -Tsvg Model.dot > Model.svg
dot -Tsvg Filter.dot > Filter.svg
dot -Tsvg Sampler.dot > Sampler.svg
dot -Tsvg Handler.dot > Handler.svg
