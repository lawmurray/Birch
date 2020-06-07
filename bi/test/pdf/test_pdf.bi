/*
 * Test the pmf of a univariate Boolean distribution.
 *
 * - π: The target distribution. 
 * - N: Number of samples.
 */
function test_pdf(π:Distribution<Boolean>, N:Integer) {
  /* simulate, counting the occurrence of each value */
  auto k <- 0;
  for n in 1..N {
    if π.simulate() {
      k <- k + 1;
    }
  }

  /* compare pdf to count */
  auto failed <- false;
  auto ε <- 5.0/sqrt(Real(N));
  
  auto δ <- abs(π.pdf(true) - Real(k)/N);
  if δ > ε {
    failed <- true;
    stderr.print("failed on true, " + δ + " > " + ε + "\n");
  }
  δ <- abs(π.pdf(false) - Real(N - k)/N);
  if δ > ε {
    failed <- true;
    stderr.print("failed on false, " + δ + " > " + ε + "\n");
  }
  if failed {
    exit(1);
  }
}

/*
 * Test the pmf of a univariate discrete distribution.
 *
 * - π: The target distribution. 
 * - N: Number of samples.
 */
function test_pdf(π:Distribution<Integer>, N:Integer) {  
  /* lower bound on interval */
  auto from <- π.lower();
  if !from? {
    from <- π.quantile(1.0e-6);
    assert from?;
  }

  /* upper bound on interval */
  auto to <- π.upper();
  if !to? {
    to <- π.quantile(1.0 - 1.0e-6);
    if !to? {
      /* search for a rough upper bound for the interval */
      auto u <- 100;
      while π.pdf(u) > 1.0e-6 {
        u <- 2*u;
      }
      to <- u;
    }
  }

  /* simulate, counting the occurrence of each value */
  auto count <- vector(0, to! - from! + 1);
  for n in 1..N {
    auto i <- π.simulate() - from! + 1;
    if 1 <= i && i <= length(count) {
      count[i] <- count[i] + 1;
    }
  }

  /* compare sum of pdf to counts */
  auto failed <- false;
  for x in from!..to! {
    auto δ <- abs(π.pdf(x) - Real(count[x - from! + 1])/N);
    auto ε <- 5.0/sqrt(Real(N));
    if δ > ε {
      failed <- true;
      stderr.print("failed on value " + x + ", " + δ + " > " + ε + "\n");
    }
  }
  if failed {
    exit(1);
  }
}

/*
 * Test a multivariate pdf.
 *
 * - π: The target distribution. 
 * - D: Number of dimensions.
 * - N: Number of samples.
 * - B: Number of samples in each batch for tuning Metropolis sampler.
 * - S: Metropolis sampler skip.
 */
function test_pdf(π:Distribution<Real[_]>, D:Integer, N:Integer, B:Integer,
    S:Integer) {  
  X1:Real[N,D];
  X2:Real[N,D];
  
  /* iid samples */
  for n in 1..N {
    X1[n,1..D] <- vector(π.simulate());
  }
  
  /* compute the shape for a Gaussian proposal using the iid samples */
  auto μ <- vector(0.0, D);
  auto Σ <- matrix(0.0, D, D);
  for n in 1..N {
    auto x <- X1[n,1..D];
    μ <- μ + x;
    Σ <- Σ + outer(x);
  }
  μ <- μ/Real(N);
  Σ <- Σ/Real(N) - outer(μ);
  
  /* scale this proposal to get a reasonable acceptance rate */
  auto done <- false;
  do {
    auto a <- 0.0;
    auto x <- π.simulate();
    auto l <- π.logpdf(x);

    for n in 1..B {
      auto x' <- simulate_multivariate_gaussian(x, Σ);
      auto l' <- π.logpdf(x');
      if log(simulate_uniform(0.0, 1.0)) <= l' - l {
        /* accept */
        x <- x';
        l <- l';
        a <- a + 1.0/B;
      }
    }
    //stderr.print("trial acceptance rate: " + a + "\n");
    if a < 0.2 {
      Σ <- 0.5*Σ;
    } else if a > 0.4 {
      Σ <- 1.5*Σ;
    } else {
      done <- true;
    }
  } while !done;

  /* apply the Metropolis sampler */
  auto a <- 0.0;
  auto x <- π.simulate();
  auto l <- π.logpdf(x);
  for n in 1..N {
    for s in 1..S {
      auto x' <- simulate_multivariate_gaussian(x, Σ);
      auto l' <- π.logpdf(x');
      if log(simulate_uniform(0.0, 1.0)) <= l' - l {
        /* accept */
        x <- x';
        l <- l';
        a <- a + 1.0/(N*S);
      }
    }
    X2[n,1..D] <- vector(x);
  }
  //stderr.print("final acceptance rate: " + a + "\n");
  
  /* test distance between the iid and Metropolis samples */
  if !pass(X1, X2) {
    exit(1);
  }
}

/*
 * Test a matrix pdf.
 *
 * - π: The target distribution. 
 * - R: Number of rows.
 * - C: Number of columns.
 * - N: Number of samples.
 * - B: Number of samples in each batch for tuning Metropolis sampler.
 * - S: Metropolis sampler skip.
 */
function test_pdf(π:Distribution<Real[_,_]>, R:Integer, C:Integer, N:Integer,
    B:Integer, S:Integer) {
  X1:Real[N,R*C];
  X2:Real[N,R*C];
  
  /* iid samples */
  for n in 1..N {
    X1[n,1..R*C] <- vector(π.simulate());
  }
  
  /* compute the shape for a Gaussian proposal using the iid samples */
  auto μ <- vector(0.0, R*C);
  auto Σ <- matrix(0.0, R*C, R*C);
  for n in 1..N {
    auto x <- X1[n,1..R*C];
    μ <- μ + x;
    Σ <- Σ + outer(x);
  }
  μ <- μ/Real(N);
  Σ <- Σ/Real(N) - outer(μ);
  
  /* scale this proposal to get a reasonable acceptance rate */
  auto done <- false;
  do {
    auto a <- 0.0;
    auto x <- π.simulate();
    auto l <- π.logpdf(x);

    for n in 1..B {
      auto x' <- matrix(simulate_multivariate_gaussian(vector(x), Σ), R, C);
      auto l' <- π.logpdf(x');
      if log(simulate_uniform(0.0, 1.0)) <= l' - l {
        /* accept */
        x <- x';
        l <- l';
        a <- a + 1.0/B;
      }
    }
    //stderr.print("trial acceptance rate: " + a + "\n");
    if a < 0.2 {
      Σ <- 0.5*Σ;
    } else if a > 0.4 {
      Σ <- 1.5*Σ;
    } else {
      done <- true;
    }
  } while !done;

  /* apply the Metropolis sampler */
  auto a <- 0.0;
  auto x <- π.simulate();
  auto l <- π.logpdf(x);
  for n in 1..N {
    for s in 1..S {
      auto x' <- matrix(simulate_multivariate_gaussian(vector(x), Σ), R, C);
      auto l' <- π.logpdf(x');
      if log(simulate_uniform(0.0, 1.0)) <= l' - l {
        /* accept */
        x <- x';
        l <- l';
        a <- a + 1.0/(N*S);
      }
    }
    X2[n,1..R*C] <- vector(x);
  }
  //stderr.print("final acceptance rate: " + a + "\n");
  
  /* test distance between the iid and Metropolis samples */
  if !pass(X1, X2) {
    exit(1);
  }
}
