/**
 * Demonstrates sampling from a triplet of Gaussian random variables, with
 * zero or more given a value.
 */
program triplet(x:(Real ~ Gaussian), y:(Real ~ Gaussian), z:(Real ~ Gaussian)) {
  x ~ Gaussian(0.0, 1.0);
  y ~ Gaussian(x, 1.0);
  z ~ Gaussian(y, 1.0);

  print("x = ");
  print(x);
  print(", y = ");
  print(y);
  print(", z = ");
  print(z);
  print("\n");
}
