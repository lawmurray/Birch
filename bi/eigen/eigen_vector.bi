program eigen_vector(N:Integer <- 5) {
  x:Real[N];
  y:Real[N];
  z:Real[N];
  
  v:Real <- 1.0;
  for (n:Integer in 1..N) {
    x[n] <- v;
    y[n] <- 2.0*v;
    v <- v + 1.0;
  }
  z <- x + y;
  
  stdout.print(z);
}
