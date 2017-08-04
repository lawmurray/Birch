program demo_fiber(N:Integer <- 10) {
  a:Integer! <- f(1, N);
  while (a?) {
    print(a!);
    print("\n");
  }
}

fiber f(from:Integer, to:Integer) -> Integer! {
  n:Integer;
  for (n in from..to) {
    yield n;
  }
}
