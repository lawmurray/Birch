/*
 * Test linear-discrete-delta conjugacy.
 */
program test_linear_discrete_delta(N:Integer <- 10000) {
  m:TestLinearDiscreteDelta;
  test_conjugacy(m, N, 2);
}
