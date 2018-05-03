/**
 * Program to run all tests.
 */
program test() {
  run_test("beta_bernoulli");
  run_test("beta_binomial");
  run_test("dirichlet_categorical");
  run_test("dirichlet_multinomial");
  run_test("gamma_poisson");
  run_test("gaussian_gaussian");
  run_test("linear_gaussian_gaussian");
  run_test("inverse_gamma_gaussian");
  run_test("normal_inverse_gamma_gaussian");
  run_test("linear_normal_inverse_gamma_gaussian");
  run_test("gaussian_log_gaussian");
  run_test("linear_gaussian_log_gaussian");
  run_test("inverse_gamma_log_gaussian");
  run_test("normal_inverse_gamma_log_gaussian");
  run_test("linear_normal_inverse_gamma_log_gaussian");
  run_test("multivariate_gaussian_gaussian");
  run_test("multivariate_linear_gaussian_gaussian");
  run_test("multivariate_inverse_gamma_gaussian");
  run_test("multivariate_normal_inverse_gamma_gaussian");
  run_test("multivariate_linear_normal_inverse_gamma_gaussian");
}

/**
 * Run a specific test.
 *
 * - test: Name of the test.
 */
function run_test(test:String) {
  tic();
  ret:Integer <- system("birch test_" + test);
  s:Real <- toc();
  if (ret == 0) {
    stdout.print("PASS");
  } else {
    stdout.print("FAIL");
  }
  stdout.print("\t" + s + "s\t");
  stdout.print(test);
  stdout.print("\n");
}
