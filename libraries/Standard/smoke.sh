#!/bin/bash
set -eo pipefail

N=10
B=10
S=0

ls src/test/basic     | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$//g"                                | bash -v
ls src/test/cdf       | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N/g"                          | bash -v
ls src/test/grad      | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N/g"                          | bash -v
ls src/test/pdf       | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N -B $B -S $S --lazy false/g" | bash -v
ls src/test/pdf       | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N -B $B -S $S --lazy true/g"  | bash -v
ls src/test/conjugacy | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N --lazy false/g"             | bash -v
ls src/test/conjugacy | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N --lazy true/g"              | bash -v
