/**
 * Demonstration of multiple dispatch.
 */
program multiple_dispatch() {
  a:A;
  b:B;
  
  print("f(a) calls ");
  f(a);
  print("\n");
  
  print("f(b) calls ");
  f(b);
  print("\n");
}

model A {
  //
}

model B < A {
  //
}

function f(a:A) {
  print("f(a:A) then g(a) calls ");
  g(a);
}

function g(a:A) {
  print("g(a:A)");
}

function g(b:B) {
  print("g(b:B)");
}
