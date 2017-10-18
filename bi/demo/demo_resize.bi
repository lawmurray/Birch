/**
 * Demonstrates how to resize an array by assignment.
 */
program demo_resize(N:Integer <- 10) {
  x:Integer[N];
  stdout.print("length is " + length(x) + "\n");
  x <- vector(0, 2*N);
  stdout.print("length is " + length(x) + "\n");
}
