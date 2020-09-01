P=4
N=100

ls bi/test/basic     | grep '\.bi' | sed "s/.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls bi/test/cdf       | grep '\.bi' | sed "s/.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls bi/test/conjugacy | grep '\.bi' | sed "s/.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls bi/test/pdf       | grep '\.bi' | sed "s/.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
