/*
 * Test analytical solution for subtraction of bounded discrete random
 * variates.
 */
program test_subtract_bounded_discrete_delta(N:Integer <- 10000) {
  m:TestSubtractBoundedDiscreteDelta;
  test_conjugacy(m, N, 2);
}
