program demo_copy_fiber(N:Integer <- 10) {
  a:Real! <- g(1, N);
  b:Real! <- a;
  
  print("1st fiber:");
  while (a?) {
    print(" ");
    print(a!);
  }
  print("\n");

  print("2nd fiber:");
  while (b?) {
    print(" ");
    print(b!);
  }
  print("\n");
}

fiber g(from:Integer, to:Integer) -> Real! {
  n:Integer;
  x:Real;
  for (n in from..to) {
    x <~ Gaussian(0.0, 1.0);
    yield x;
  }
}
