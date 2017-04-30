program eigen_vector(N:Integer <- 5) {
  x:Real[N];
  y:Real[N];
  z:Real[N];
  n:Integer;
  
  for (n in 1..N) {
    x[n] <- 1.0;
    y[n] <- 2.0;
  }
  z <- x + y;
  
  for (n in 1..N) {
    print(z[n]);
    print("\n");
  }
}
