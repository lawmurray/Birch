/**
 * Run all tests.
 */
program test(N:Integer <- 10000) {
  code:Integer <- 0;

  code <- code + run_test("deep_clone_alias");
  code <- code + run_test("deep_clone_chain");
  code <- code + run_test("deep_clone_modify_dst");
  code <- code + run_test("deep_clone_modify_src");
  code <- code + run_test("cdf_beta");
  code <- code + run_test("cdf_beta_binomial");
  code <- code + run_test("cdf_binomial");
  code <- code + run_test("cdf_exponential");
  code <- code + run_test("cdf_gamma");
  code <- code + run_test("cdf_gamma_poisson");
  code <- code + run_test("cdf_gaussian");
  code <- code + run_test("cdf_geometric");
  code <- code + run_test("cdf_inverse_gamma");
  code <- code + run_test("cdf_negative_binomial");
  code <- code + run_test("cdf_poisson");
  code <- code + run_test("cdf_student_t");
  code <- code + run_test("cdf_weibull");
  code <- code + run_test("pdf_multivariate_gaussian");
  code <- code + run_test("add_bounded_discrete_delta", N);
  code <- code + run_test("beta_bernoulli", N);
  code <- code + run_test("beta_binomial", N);
  code <- code + run_test("beta_geometric", N);
  code <- code + run_test("beta_negative_binomial", N);
  code <- code + run_test("chain_gaussian", N);
  code <- code + run_test("chain_multivariate_gaussian", N);
  code <- code + run_test("dirichlet_categorical", N);
  code <- code + run_test("dirichlet_multinomial", N);
  code <- code + run_test("gamma_exponential", N);
  code <- code + run_test("gamma_poisson", N);
  code <- code + run_test("gaussian_gaussian", N);
  code <- code + run_test("inverse_gamma_gamma", N);
  code <- code + run_test("linear_discrete_delta", N);
  code <- code + run_test("linear_gaussian_gaussian", N);
  code <- code + run_test("linear_matrix_normal_inverse_gamma_matrix_gaussian", N);
  code <- code + run_test("linear_matrix_normal_inverse_wishart_matrix_gaussian", N);
  code <- code + run_test("linear_multivariate_gaussian_multivariate_gaussian", N);
  code <- code + run_test("linear_multivariate_normal_inverse_gamma_multivariate_gaussian", N);
  code <- code + run_test("linear_normal_inverse_gamma_gaussian", N);
  code <- code + run_test("matrix_normal_inverse_gamma_matrix_gaussian", N);
  code <- code + run_test("matrix_normal_inverse_wishart_matrix_gaussian", N);
  code <- code + run_test("multivariate_gaussian_multivariate_gaussian", N);
  code <- code + run_test("multivariate_normal_inverse_gamma_multivariate_gaussian", N);
  code <- code + run_test("normal_inverse_gamma", N);
  code <- code + run_test("normal_inverse_gamma_gaussian", N);
  code <- code + run_test("scaled_gamma_exponential", N);
  code <- code + run_test("scaled_gamma_poisson", N);
  code <- code + run_test("subtract_bounded_discrete_delta", N);
  
  exit(code);
}

/*
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

/*
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

/*
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
  
  auto R <- rows(X1);
  auto C <- columns(X1);
  auto failed <- 0;
  auto tests <- 0;
  auto ε <- 5.0/sqrt(R);
  
  /* compare marginals using 1-Wasserstein distance */
  for auto c in 1..C {
    /* project onto a random unit vector for univariate comparison */
    auto x1 <- X1[1..R,c];
    auto x2 <- X2[1..R,c];
  
    /* normalise onto the interval [0,1] */
    auto mn <- min(min(x1), min(x2));
    auto mx <- max(max(x1), max(x2));
    auto z1 <- (x1 - mn)/(mx - mn);
    auto z2 <- (x2 - mn)/(mx - mn);
    
    /* compute distance and suggested pass threshold */
    auto δ <- wasserstein(z1, z2);
    if δ > ε {
      failed <- failed + 1;
      stderr.print("failed on component " + c + ", " + δ + " > " + ε + "\n");
    }
    tests <- tests + 1;
  }
  
  /* project onto random unit vectors and compute the univariate
   * 1-Wasserstein distance, repeating as many times as there are
   * dimensions */
  for auto c in 1..C {
    /* project onto a random unit vector for univariate comparison */
    auto u <- simulate_uniform_unit_vector(C);  
    auto x1 <- X1*u;
    auto x2 <- X2*u;
  
    /* normalise onto the interval [0,1] */
    auto mn <- min(min(x1), min(x2));
    auto mx <- max(max(x1), max(x2));
    auto z1 <- (x1 - mn)/(mx - mn);
    auto z2 <- (x2 - mn)/(mx - mn);
  
    /* compute distance and suggested pass threshold */
    auto δ <- wasserstein(z1, z2);    
    if δ > ε {
      failed <- failed + 1;
      stderr.print("failed on random projection, " + δ + " > " + ε + "\n");
    }
    tests <- tests + 1;
  }
  
  if failed > 0 {
    stderr.print("failed " + failed + " of " + tests + " comparisons\n");
  }
  return failed == 0;
}
