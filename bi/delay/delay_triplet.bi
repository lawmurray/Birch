/**
 * Demonstrates sampling from a triplet of Gaussian random variables, with
 * zero or more given a value.
 */
program delay_triplet(x:(Real ~ Gaussian), y:(Real ~ Gaussian), z:(Real ~ Gaussian)) {
  x ~ Gaussian(0.0, 1.0);
  y ~ Gaussian(x, 1.0);
  z ~ Gaussian(10.0*y, 10.0);
  
  print("x = ");
  print(x);
  print("\ny = ");
  print(y);
  print("\nz = ");
  print(z);
  print("\n");
}
