program demo_copy_fiber(N:Integer <- 10) {
  a:Real! <- g(1, N);
  b:Real! <- a;
  
  stdout.print("1st fiber:");
  while (a?) {
    stdout.printf(" %f", a!);
  }
  stdout.print("\n");

  stdout.print("2nd fiber:");
  while (b?) {
    stdout.printf(" %f", b!);
  }
  stdout.print("\n");
}

fiber g(from:Integer, to:Integer) -> Real! {
  n:Integer;
  x:Real;
  for (n in from..to) {
    x <~ Gaussian(0.0, 1.0);
    yield x;
  }
}
