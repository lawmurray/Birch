/*
 * Test the pmf of a univariate Boolean distribution.
 *
 * - π: The target distribution. 
 * - N: Number of samples.
 */
function test_pdf(π:Distribution<Boolean>, N:Integer) {  
  /* simulate, counting the occurrence of each value */
  auto k <- 0;
  for auto n in 1..N {
    if π.simulate() {
      k <- k + 1;
    }
  }

  /* compare pdf to count */
  auto failed <- false;
  auto ε <- 5.0/sqrt(N);
  
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
    assert to?;
  }

  /* simulate, counting the occurrence of each value */
  auto count <- vector(0, to! - from! + 1);
  for auto n in 1..N {
    auto i <- π.simulate() - from! + 1;
    count[i] <- count[i] + 1;
  }

  /* compare sum of pdf to counts */
  auto failed <- false;
  for auto x in from!..to! {
    auto δ <- abs(π.pdf(x) - Real(count[x - from! + 1])/N);
    auto ε <- 5.0/sqrt(N);
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
  for auto n in 1..N {
    X1[n,1..D] <- π.simulate();
  }
  
  /* tune the proposal distribution for a Metropolis sampler */
  auto a <- 0.0;  // acceptance rate
  auto μ <- vector(0.0, D);
  auto Σ <- identity(D);
  while a < 0.2 || a > 0.4 {
    auto q <- Gaussian(μ, Σ);  // proposal
    auto x <- π.simulate();
    a <- 0.0;
    for auto n in 1..B {
      auto x' <- x + q.simulate();
      if simulate_uniform(0.0, 1.0) <= exp(π.logpdf(x') - π.logpdf(x)) {
        x <- x';  // accept
        a <- a + 1.0/B;
      }
    }
    if a < 0.2 {
      Σ <- 0.5*Σ;
    } else if a > 0.4 {
      Σ <- 2.0*Σ;
    }
    stderr.print("trial acceptance rate = " + a + "\n");
  }

  /* apply the Metropolis sampler */
  auto q <- Gaussian(μ, Σ);  // proposal
  auto x <- π.simulate();
  a <- 0.0;
  for auto n in 1..N {
    for auto s in 1..S {
      auto x' <- x + q.simulate();
      if simulate_uniform(0.0, 1.0) <= exp(π.logpdf(x') - π.logpdf(x)) {
        x <- x';  // accept
        a <- a + 1.0/(N*S);
      }
    }
    X2[n,1..D] <- x;
  }  
  stderr.print("final acceptance rate = " + a + "\n");
  
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
  for auto n in 1..N {
    X1[n,1..R*C] <- vector(π.simulate());
  }
  
  /* tune the proposal distribution for a Metropolis sampler */
  auto x <- π.simulate();
  auto a <- 0.0;  // acceptance rate
  auto μ <- vector(0.0, R*C);
  auto Σ <- identity(R*C);
  while a < 0.2 || a > 0.4 {
    auto q <- Gaussian(μ, Σ);  // proposal
    a <- 0.0;
    for auto n in 1..B {
      auto x' <- x + matrix(q.simulate(), rows(x), columns(x));
      if simulate_uniform(0.0, 1.0) <= exp(π.logpdf(x') - π.logpdf(x)) {
        x <- x';  // accept
        a <- a + 1.0/B;
      }
    }
    if a < 0.2 {
      Σ <- 0.5*Σ;
    } else if a > 0.4 {
      Σ <- 2.0*Σ;
    }
    stderr.print("trial acceptance rate = " + a + "\n");
  }

  /* apply the Metropolis sampler */
  auto q <- Gaussian(μ, Σ);  // proposal
  a <- 0.0;
  for auto n in 1..N {
    for auto s in 1..S {
      auto x' <- x + matrix(q.simulate(), rows(x), columns(x));
      if simulate_uniform(0.0, 1.0) <= exp(π.logpdf(x') - π.logpdf(x)) {
        x <- x';  // accept
        a <- a + 1.0/(N*S);
      }
    }
    X2[n,1..R*C] <- vector(x);
  }  
  stderr.print("final acceptance rate = " + a + "\n");
  
  /* test distance between the iid and Metropolis samples */
  if !pass(X1, X2) {
    exit(1);
  }
}
