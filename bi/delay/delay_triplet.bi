/**
 * Demonstrates sampling from a triplet of Gaussian random variables, with
 * zero or more of them given a value on the command line.
 *
 *   - `-x`            : Value of the first variable.
 *   - `-y`            : Value of the second variable.
 *   - `-z`            : Value of the third variable.
 */
program delay_triplet(x:Real?, y:Real?, z:Real?) {
  x1:Random<Real>;
  y1:Random<Real>;
  z1:Random<Real>;
  
  x1 <- x;
  y1 <- y;
  z1 <- z;
  
  x1 ~ Gaussian(0.0, 1.0);
  y1 ~ Gaussian(x1, 1.0);
  z1 ~ Gaussian(y1, 1.0);
  
  stdout.print("x = " + x1 + "\n");
  stdout.print("y = " + y1 + "\n");
  stdout.print("z = " + z1 + "\n");
}
