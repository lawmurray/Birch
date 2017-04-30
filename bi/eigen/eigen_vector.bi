program eigen_vector(N:Integer <- 5) {
  x:Real[N];
  y:Real[N];
  z:Real[N];
  n:Integer;
  
  v:Real <- 1.0;
  for (n in 1..N) {
    x[n] <- v;
    y[n] <- 2.0*v;
    v <- v + 1.0;
  }
  z <- x + y;
  
  for (n in 1..N) {
    print(z[n]);
    print("\n");
  }
}
