program demo_member_fiber(N:Integer <- 10) {
  a:A(1, N);
  b:Real! <- a.f();
  while (b?) {
    stdout.printf("%f\n", b!);
  }
}

class A(from:Integer, to:Integer) {
  fiber f() -> Real! {
    n:Integer;
    x:Real;
    for (n in from..to) {
      x <~ Gaussian(0.0, 1.0);
      yield x;
    }
  }
}
