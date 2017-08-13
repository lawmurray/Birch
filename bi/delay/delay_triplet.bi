/**
 * Demonstrates sampling from a triplet of Gaussian random variables, with
 * zero or more of them given a value on the command line.
 *
 *   - x : Value of the first variable.
 *   - y : Value of the second variable.
 *   - z : Value of the third variable.
 */
program delay_triplet(x:Gaussian, y:Gaussian, z:Gaussian) {  
  x ~ Gaussian(0.0, 1.0);
  y ~ Gaussian(x, 1.0);
  z ~ Gaussian(y, 1.0);
  
  print("x = ");
  print(x);
  print("\ny = ");
  print(y);
  print("\nz = ");
  print(z);
  print("\n");
}
