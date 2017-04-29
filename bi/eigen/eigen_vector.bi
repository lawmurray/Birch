program eigen_vector(N:Integer <- 5) {
  x:Real[N];
  y:Real[N];
  z:Real[N];
  n:Integer;
  
  for (n in 1..N) {
    x[n] <- n;
    y[n] <- 2*n;
  }
  z <- x + y;
  
  for (n in 1..N) {
    print(z[n]);
    print("\n");
  }
}
