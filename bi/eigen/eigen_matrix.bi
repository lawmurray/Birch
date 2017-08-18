program eigen_matrix(R:Integer <- 3, C:Integer <- 3) {
  X:Real[R,C];
  Y:Real[R,C];
  Z:Real[R,C];
  i:Integer;
  j:Integer;
  
  v:Real <- 1.0;
  for (i in 1..R) {
    for (j in 1..C) {
      X[i,j] <- v;
      Y[i,j] <- 2.0*v;
      v <- v + 1.0;
    }
  }
  Z <- X + Y;
  
  stdout.print(Z);
}
