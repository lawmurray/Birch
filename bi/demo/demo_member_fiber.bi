program demo_member_fiber(N:Integer <- 10) {
  a:A(1, N);
  b:Real! <- a.f();
  while (b?) {
    stdout.print(b! + "\n");
  }
}

class A(from:Integer, to:Integer) {
  from:Integer <- from;
  to:Integer <- to;

  fiber f() -> Real! {
    for (n:Integer in from..to) {
      yield simulate_gaussian(0.0, 1.0);
    }
  }
}
