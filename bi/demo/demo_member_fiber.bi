program demo_member_fiber(N:Integer <- 10) {
  a:A(1, N);
  b:Integer! <- a.f();
  while (b?) {
    print(b!);
    print("\n");
  }
}

class A(from:Integer, to:Integer) {
  fiber f() -> Integer! {
    n:Integer;
    for (n in from..to) {
      yield n;
    }
  }
}
