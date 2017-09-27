program demo_fiber(N:Integer <- 10) {
  a:Real! <- f(1, N);
  while (a?) {
    stdout.print(a! + "\n");
  }
}

fiber f(from:Integer, to:Integer) -> Real! {
  for (n:Integer in from..to) {
    yield simulate_gaussian(0.0, 1.0);
  }
}
