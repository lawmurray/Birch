/*
 * Test linear-discrete-delta conjugacy.
 */
program test_negative_linear_discrete_delta(N:Integer <- 10000) {
  m:TestNegativeLinearDiscreteDelta;
  test_conjugacy(m, N, 2);
}
