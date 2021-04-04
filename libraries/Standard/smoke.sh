#!/bin/bash
set -eo pipefail

M=10   # for gradient tests
N=10   # for sampling tests
B=10
S=0

ls src/test/basic     | grep '\.birch' | sed "s/.birch$/ -N $N/g"              | xargs -t -L 1 birch
ls src/test/cdf       | grep '\.birch' | sed "s/.birch$/ -N $N/g"              | xargs -t -L 1 birch
ls src/test/grad      | grep '\.birch' | sed "s/.birch$/ -N $M/g"              | xargs -t -L 1 birch
ls src/test/pdf       | grep '\.birch' | sed "s/.birch$/ -N $N -B $B -S $S --lazy false/g" | xargs -t -L 1 birch
ls src/test/pdf       | grep '\.birch' | sed "s/.birch$/ -N $N -B $B -S $S --lazy true/g"  | xargs -t -L 1 birch
ls src/test/conjugacy | grep '\.birch' | sed "s/.birch$/ -N $N --lazy false/g" | xargs -t -L 1 birch
ls src/test/conjugacy | grep '\.birch' | sed "s/.birch$/ -N $N --lazy true/g"  | xargs -t -L 1 birch
