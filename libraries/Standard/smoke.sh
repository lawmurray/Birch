#!/bin/bash
set -eo pipefail

P=4
M=10   # for gradient tests
N=100  # for sampling tests

ls src/test/basic     | grep '\.birch' | sed "s/.birch$/ -N $N/g"              | xargs -t -L 1 -P $P birch
ls src/test/cdf       | grep '\.birch' | sed "s/.birch$/ -N $N/g"              | xargs -t -L 1 -P $P birch
ls src/test/grad      | grep '\.birch' | sed "s/.birch$/ -N $M/g"              | xargs -t -L 1 -P $P birch
ls src/test/pdf       | grep '\.birch' | sed "s/.birch$/ -N $N --lazy false/g" | xargs -t -L 1 -P $P birch
ls src/test/pdf       | grep '\.birch' | sed "s/.birch$/ -N $N --lazy true/g"  | xargs -t -L 1 -P $P birch
ls src/test/conjugacy | grep '\.birch' | sed "s/.birch$/ -N $N --lazy false/g" | xargs -t -L 1 -P $P birch
ls src/test/conjugacy | grep '\.birch' | sed "s/.birch$/ -N $N --lazy true/g"  | xargs -t -L 1 -P $P birch
