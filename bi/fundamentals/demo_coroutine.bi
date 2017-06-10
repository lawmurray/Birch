program demo_coroutine(N:Integer <- 10) {
  c:Coroutine<Integer> <- f(N);
  n:Integer;
  for (n in 1..N) {
    print(c());
    print("\n");
  }
}

coroutine f(N:Integer) -> Integer {
  n:Integer;
  for (n in 1..N) {
    return n;
  }
}
