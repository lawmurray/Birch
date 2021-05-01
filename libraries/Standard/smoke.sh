#!/bin/bash
#set -eov pipefail
set -v

N=4
B=4
S=0

eval "`ls src/test/basic     | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$//g"`"
eval "`ls src/test/cdf       | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N/g"`"
eval "`ls src/test/grad      | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N --backward false/g"`"
eval "`ls src/test/grad      | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N --backward true/g"`"
eval "`ls src/test/pdf       | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N -B $B -S $S --lazy false/g"`"
eval "`ls src/test/pdf       | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N -B $B -S $S --lazy true/g"`"
eval "`ls src/test/conjugacy | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N --lazy false/g"`"
eval "`ls src/test/conjugacy | grep '\.birch' | sed "s/^/birch /g" | sed "s/.birch$/ -N $N --lazy true/g"`"
