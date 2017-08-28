program demo_fiber(N:Integer <- 10) {
  a:Real! <- f(1, N);
  while (a?) {
    stdout.printf("%f\n", a!);
  }
}

fiber f(from:Integer, to:Integer) -> Real! {
  n:Integer;
  for (n in from..to) {
    yield random_gaussian(0.0, 1.0);
  }
}
