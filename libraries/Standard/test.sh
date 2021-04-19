#!/bin/bash
set -eo pipefail

N1=100000   # for cdf tests
N2=1000     # for gradient tests
N3=10000    # for pdf tests
N4=100000  # for conjugacy tests

ls src/test/basic     | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$//g"                     | bash -v
ls src/test/cdf       | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N1/g"              | bash -v
ls src/test/grad      | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N2/g"              | bash -v
ls src/test/pdf       | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N3 --lazy false/g" | bash -v
ls src/test/pdf       | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N3 --lazy true/g"  | bash -v
ls src/test/conjugacy | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N4 --lazy false/g" | bash -v
ls src/test/conjugacy | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N4 --lazy true/g"  | bash -v
