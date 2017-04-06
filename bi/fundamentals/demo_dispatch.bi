/**
 * Demonstration of runtime multiple dispatch.
 */
program demo_dispatch() {
  a:A;
  b:B;

  f(a);  // calls g(A) via f(A), determined at runtime
  f(b);  // calls g(B) via f(A), determined at runtime
}

model A {
  //
}

model B < A {
  //
}

function f(a:A) {
  /* a is treated as being of type A here, but calling g(a) will resolve to
   * g(A) or g(B) according to the specific type of a, determined at
   * runtime */
  g(a);
}

function g(a:A) {
  print("called g(a:A)\n");
}

function g(b:B) {
  print("called g(b:B)\n");
}
