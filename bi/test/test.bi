/**
 * Program to run all tests.
 */
program test() {
  code:Integer <- 0;
  
  code <- code + run_test("beta_bernoulli");
  code <- code + run_test("beta_binomial");
  code <- code + run_test("dirichlet_categorical");
  code <- code + run_test("dirichlet_multinomial");
  code <- code + run_test("gamma_poisson");
  code <- code + run_test("linear_discrete_delta");
  code <- code + run_test("gaussian_gaussian");
  code <- code + run_test("linear_gaussian_gaussian");
  code <- code + run_test("inverse_gamma_gaussian");
  code <- code + run_test("normal_inverse_gamma_gaussian");
  code <- code + run_test("linear_normal_inverse_gamma_gaussian");
  code <- code + run_test("gaussian_log_gaussian");
  code <- code + run_test("linear_gaussian_log_gaussian");
  code <- code + run_test("inverse_gamma_log_gaussian");
  code <- code + run_test("normal_inverse_gamma_log_gaussian");
  code <- code + run_test("linear_normal_inverse_gamma_log_gaussian");
  code <- code + run_test("multivariate_gaussian_gaussian");
  code <- code + run_test("multivariate_linear_gaussian_gaussian");
  code <- code + run_test("multivariate_inverse_gamma_gaussian");
  code <- code + run_test("multivariate_normal_inverse_gamma_gaussian");
  code <- code + run_test("multivariate_linear_normal_inverse_gamma_gaussian");
  
  exit(code);
}

/**
 * Run a specific test.
 *
 * - test: Name of the test.
 *
 * Return: Exit code of the test.
 */
function run_test(test:String) -> Integer {
  tic();
  code:Integer <- system("birch test_" + test);
  s:Real <- toc();
  if (code == 0) {
    stdout.print("PASS");
  } else {
    stdout.print("FAIL");
  }
  stdout.print("\t" + s + "s\t");
  stdout.print(test);
  stdout.print("\n");
  
  return code;
}
