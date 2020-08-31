P=4
N=100000

ls bi/basic     | grep '\.bi' | sed "s/.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls bi/cdf       | grep '\.bi' | sed "s/.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls bi/conjugacy | grep '\.bi' | sed "s/.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls bi/pdf       | grep '\.bi' | sed "s/.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
