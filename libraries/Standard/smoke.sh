P=4
N=100

ls src/test/basic     | grep '\.birch' | sed "s/.birch$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls src/test/cdf       | grep '\.birch' | sed "s/.birch$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls src/test/conjugacy | grep '\.birch' | sed "s/.birch$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls src/test/pdf       | grep '\.birch' | sed "s/.birch$/ -N $N/g" | xargs -t -L 1 -P $P birch
