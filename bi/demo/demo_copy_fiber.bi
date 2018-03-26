program demo_copy_fiber(N:Integer <- 10) {
  a:Real! <- g(1, N);
  b:Real! <- a;
  
  stdout.print("1st fiber:");
  while (a?) {
    stdout.print(" " + a!);
  }
  stdout.print("\n");

  stdout.print("2nd fiber:");
  while (b?) {
    stdout.print(" " + b!);
  }
  stdout.print("\n");
}

fiber g(from:Integer, to:Integer) -> Real {
  for (n:Integer in from..to) {
    yield simulate_gaussian(0.0, 1.0);
  }
}
