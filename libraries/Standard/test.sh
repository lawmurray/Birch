P=4
N=100000

ls birch/test/basic     | grep '\.bi' | sed "s/.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls birch/test/cdf       | grep '\.bi' | sed "s/.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls birch/test/conjugacy | grep '\.bi' | sed "s/.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls birch/test/pdf       | grep '\.bi' | sed "s/.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
