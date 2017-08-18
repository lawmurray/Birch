program demo_fiber(N:Integer <- 10) {
  a:Real! <- f(1, N);
  while (a?) {
    stdout.printf("%f\n", a!);
  }
}

fiber f(from:Integer, to:Integer) -> Real! {
  n:Integer;
  x:Real;
  for (n in from..to) {
    x <~ Gaussian(0.0, 1.0);
    yield x;
  }
}
