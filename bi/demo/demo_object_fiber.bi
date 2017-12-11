/**
 * Demonstrates how a closed fiber yields an object by value.
 */
program demo_object_fiber(N:Integer <- 10) {
  a:Counter! <- g(N);
  while (a?) {
    a!.count <- 0; // should not reset count, as closed fiber yields by value
  }
}

closed fiber g(N:Integer) -> Counter! {
  counter:Counter;
  for (n:Integer in 1..N) {
    counter.count <- counter.count + 1;
    stdout.print(counter.count + "\n");
    yield counter;
  }
}

class Counter {
  count:Integer <- 0;
}
