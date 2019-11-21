#!/bin/sh

dot -Tsvg:cairo MarkovModel.dot > MarkovModel.svg
dot -Tsvg:cairo HiddenMarkovModel.dot > HiddenMarkovModel.svg
dot -Tsvg:cairo Model.dot > Model.svg
dot -Tsvg:cairo Sampler.dot > Sampler.svg
dot -Tsvg:cairo Handler.dot > Handler.svg
