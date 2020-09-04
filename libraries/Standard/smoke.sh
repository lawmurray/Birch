P=4
N=100

ls birch/test/basic     | grep '\.birch' | sed "s/.birch$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls birch/test/cdf       | grep '\.birch' | sed "s/.birch$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls birch/test/conjugacy | grep '\.birch' | sed "s/.birch$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls birch/test/pdf       | grep '\.birch' | sed "s/.birch$/ -N $N/g" | xargs -t -L 1 -P $P birch
