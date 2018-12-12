/**
 * Run all tests.
 */
program test(N:Integer <- 10000) {
  code:Integer <- 0;
  
  code <- code + run_test("deep_clone_alias");
  code <- code + run_test("deep_clone_chain");
  code <- code + run_test("deep_clone_modify_dst");
  code <- code + run_test("deep_clone_modify_src");
  code <- code + run_test("beta_bernoulli", N);
  code <- code + run_test("beta_binomial", N);
  code <- code + run_test("dirichlet_categorical", N);
  code <- code + run_test("dirichlet_multinomial", N);
  code <- code + run_test("gamma_poisson", N);
  code <- code + run_test("gamma_exponential", N);
  code <- code + run_test("scaled_gamma_poisson", N);
  code <- code + run_test("linear_discrete_delta", N);
  code <- code + run_test("add_bounded_discrete_delta", N);
  code <- code + run_test("subtract_bounded_discrete_delta", N);
  code <- code + run_test("gaussian_gaussian", N);
  code <- code + run_test("linear_gaussian_gaussian", N);
  code <- code + run_test("chain_gaussian", N);
  code <- code + run_test("inverse_gamma_gaussian", N);
  code <- code + run_test("normal_inverse_gamma_gaussian", N);
  code <- code + run_test("linear_normal_inverse_gamma_gaussian", N);
  code <- code + run_test("gaussian_log_gaussian", N);
  code <- code + run_test("linear_gaussian_log_gaussian", N);
  code <- code + run_test("inverse_gamma_log_gaussian", N);
  code <- code + run_test("normal_inverse_gamma_log_gaussian", N);
  code <- code + run_test("linear_normal_inverse_gamma_log_gaussian", N);
  code <- code + run_test("multivariate_gaussian_gaussian", N);
  code <- code + run_test("multivariate_linear_gaussian_gaussian", N);
  code <- code + run_test("multivariate_chain_gaussian", N);
  code <- code + run_test("multivariate_inverse_gamma_gaussian", N);
  code <- code + run_test("multivariate_normal_inverse_gamma_gaussian", N);
  code <- code + run_test("multivariate_linear_normal_inverse_gamma_gaussian", N);
  
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

/**
 * Run a specific test, where a number of samples is required.
 *
 * - test: Name of the test.
 * - N: Number of samples.
 *
 * Return: Exit code of the test.
 */
function run_test(test:String, N:Integer) -> Integer {
  tic();
  code:Integer <- system("birch test_" + test + " -N " + N);
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

/**
 * Compare two empirical distributions for the purposes of tests.
 *
 * - X1: First empirical distribution.
 * - X2: Second empirical distribution.
 *
 * Return: Did the test pass?
 */
function pass(X1:Real[_,_], X2:Real[_,_]) -> Boolean {
  assert rows(X1) == rows(X2);
  assert columns(X1) == columns(X2);
  
  R:Integer <- rows(X1);
  C:Integer <- columns(X1);
  
  /* project onto a random unit vector for univariate comparison */
  u:Real[_] <- simulate_uniform_unit_vector(C);  
  x1:Real[_] <- X1*u;
  x2:Real[_] <- X2*u;
  
  /* normalise onto the interval [0,1] */
  mn:Real <- min(min(x1), min(x2));
  mx:Real <- max(max(x1), max(x2));
  x1 <- (x1 - mn)/(mx - mn);
  x2 <- (x2 - mn)/(mx - mn);
  
  /* compute distance and suggested pass threshold */
  δ:Real <- wasserstein(x1, x2);
  ε:Real <- 1.0/sqrt(R);
  
  //stderr.print(δ + " vs " + ε + "\n");
  
  return δ < ε;
}
