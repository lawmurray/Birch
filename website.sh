#!/bin/sh

cp README.md docs/index.md
mkdir -p docs/libraries

for lib in Standard Cairo SQLite; do
  cd libraries/$lib && birch docs && cd ../..
  cp -r libraries/$lib/docs docs/libraries/$lib
done
