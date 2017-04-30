program eigen_matrix(R:Integer <- 3, C:Integer <- 3) {
  X:Real[R,C];
  Y:Real[R,C];
  Z:Real[R,C];
  i:Integer;
  j:Integer;
  
  for (i in 1..R) {
    for (j in 1..C) {
      X[i,j] <- 1.0;
      Y[i,j] <- 2.0;
    }
  }
  Z <- X + Y;
  
  for (i in 1..R) {
    for (j in 1..C) {
      print(Z[i,j]);
      print(" ");
    }
    print("\n");
  }
}
