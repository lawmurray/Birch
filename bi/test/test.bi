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
  run_test("affine_gaussian_gaussian");
  run_test("inverse_gamma_gaussian");
  run_test("normal_inverse_gamma_gaussian");
  run_test("affine_normal_inverse_gamma_gaussian");
}

/**
 * Run a specific test.
 *
 * - test: Name of the test.
 */
function run_test(test:String) {
  ret:Integer <- system("birch test_" + test);
  if (ret == 0) {
    stdout.print("PASS");
  } else {
    stdout.print("FAIL");
  }
  stdout.print("    ");
  stdout.print(test);
  stdout.print("\n");
}
