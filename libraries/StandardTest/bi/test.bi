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
  auto ε <- 4.0*columns(X1)/sqrt(Real(R));

  /* compare marginals using 1-Wasserstein distance */
  for c in 1..C {
    /* project onto a random unit vector for univariate comparison */
    auto x1 <- X1[1..R,c];
    auto x2 <- X2[1..R,c];

    /* normalise onto the interval [0,1] */
    auto mn <- min(min(x1), min(x2));
    auto mx <- max(max(x1), max(x2));
    auto z1 <- (x1 - vector(mn, R))/(mx - mn);
    auto z2 <- (x2 - vector(mn, R))/(mx - mn);

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
  for c in 1..C {
    /* project onto a random unit vector for univariate comparison */
    auto u <- simulate_uniform_unit_vector(C);
    auto x1 <- X1*u;
    auto x2 <- X2*u;

    /* normalise onto the interval [0,1] */
    auto mn <- min(min(x1), min(x2));
    auto mx <- max(max(x1), max(x2));
    auto z1 <- (x1 - vector(mn, R))/(mx - mn);
    auto z2 <- (x2 - vector(mn, R))/(mx - mn);

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
