#!/bin/sh

dot -Tsvg:cairo MarkovModel.dot > MarkovModel.svg
dot -Tsvg:cairo StateSpaceModel.dot > StateSpaceModel.svg
dot -Tsvg:cairo StarModel.dot > StarModel.svg
