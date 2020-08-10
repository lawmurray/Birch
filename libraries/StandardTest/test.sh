P=4
N=100000

ls bi/basic     | sed "s/\.bi\$//g"      | xargs -t -L 1 -P $P birch
ls bi/cdf       | sed "s/\.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls bi/conjugacy | sed "s/\.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
ls bi/pdf       | sed "s/\.bi$/ -N $N/g" | xargs -t -L 1 -P $P birch
