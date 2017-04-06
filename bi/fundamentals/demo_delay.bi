/**
 * Demonstrates the life cycle of a delayed variate, first behaving as a
 * distribution, then, after delayed sampling, behaving as a variate.
 */
program demo_delay() {
  x:(Real ~ Gaussian);
  
  /* construct the delayed variate */
  x ~ Gaussian(0.0, 1.0);
  
  /* the delayed variate may be treated as if of type Gaussian at this
   * point, so f(x) will resolve to f(Gaussian) */
  f(x);
  
  /* g(x) is only overloaded for the Real type, so calling g(x) forces
   * instantiation of the delayed variate */
  g(x);
  
  /* the delayed variate is now instantiated, and may only be treated as if
   * of type Real from this point on, so f(x) will resolve to
   * f(Real) */
  f(x);
  
  print("x = ");
  print(x);
  print("\n");
}

function f(x:Real) {
  print("called f(Real)\n");
}

function f(x:Gaussian) {
  print("called f(Gaussian)\n");
}

/**
 * An arbitrary function that is only overloaded for the variate type.
 * Calling it with a delayed variate argument will force the instanation of
 * the variate.
 */
function g(x:Real) {
  print("called g(Real)\n");
}
