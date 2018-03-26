/**
 * Demonstration of generic type parameters in a class.
 */
program demo_generic() {
  x:DemoGeneric<Gaussian>;
}

class DemoGeneric<T <= Gaussian> {
  a:T;
  
  function get() -> T {
    return a;
  }
  
  fiber run() -> T {
    yield a;
  }
}

class DemoA {
  //
}
