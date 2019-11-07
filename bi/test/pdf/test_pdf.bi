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
 * - B: Metropolis sampler burn-in.
 * - S: Metropolis sampler skip.
 */
function test_pdf(π:Distribution<Real[_]>, D:Integer, N:Integer, B:Integer,
    S:Integer) {
  auto q <- Gaussian(vector(0.0, D), diagonal(0.01, D));
  
  X1:Real[N,D];
  X2:Real[N,D];
  
  /* iid samples */
  for auto n in 1..N {
    X1[n,1..D] <- π.simulate();
  }
  
  /* Metropolis sampler, initialize */
  auto x <- π.simulate();
  
  /* burn in */
  auto μ <- vector(0.0, D);
  auto Σ <- matrix(0.0, D, D);
  for auto b in 1..B {
    auto x' <- x + q.simulate();
    if simulate_uniform(0.0, 1.0) <= π.pdf(x')/π.pdf(x) {
      x <- x';  // accept
    }
    μ <- μ + x;
    Σ <- Σ + x*transpose(x);
  }
  
  /* adapt */
  μ <- μ/N;
  Σ <- Σ/N - μ*transpose(μ);
  q <- Gaussian(vector(0.0, D), (pow(2.4, 2.0)/D)*Σ);

  /* sample */
  auto A <- 0;
  for auto n in 1..N {
    for auto s in 1..S {
      auto x' <- x + q.simulate();
      if simulate_uniform(0.0, 1.0) <= π.pdf(x')/π.pdf(x) {
        x <- x';  // accept
        A <- A + 1;
      }
    }
    X2[n,1..D] <- x;
  }  
  //stderr.print("acceptance rate = " + Real(A)/(N*S) + "\n");
  
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
 * - B: Metropolis sampler burn-in.
 * - S: Metropolis sampler skip.
 */
function test_pdf(π:Distribution<Real[_,_]>, R:Integer, C:Integer, N:Integer,
    B:Integer, S:Integer) {
  auto q <- Gaussian(vector(0.0, R*C), diagonal(0.01, R*C));
  
  X1:Real[N,R*C];
  X2:Real[N,R*C];
  
  /* iid samples */
  for auto n in 1..N {
    X1[n,1..R*C] <- vector(π.simulate());
  }
  
  /* Metropolis sampler, initialize */
  auto x <- vector(π.simulate());
  
  /* burn in */
  auto μ <- vector(0.0, R*C);
  auto Σ <- matrix(0.0, R*C, R*C);
  for auto b in 1..B {
    auto x' <- x + q.simulate();
    if simulate_uniform(0.0, 1.0) <= π.pdf(matrix(x', R, C))/π.pdf(matrix(x, R, C)) {
      x <- x';  // accept
    }
    μ <- μ + x;
    Σ <- Σ + x*transpose(x);
  }
  
  /* adapt */
  μ <- μ/N;
  Σ <- Σ/N - μ*transpose(μ);
  q <- Gaussian(vector(0.0, R*C), (pow(2.4, 2.0)/(R*C))*Σ);

  /* sample */
  auto A <- 0;
  for auto n in 1..N {
    for auto s in 1..S {
      auto x' <- x + q.simulate();
      if simulate_uniform(0.0, 1.0) <= π.pdf(matrix(x', R, C))/π.pdf(matrix(x, R, C)) {
        x <- x';  // accept
        A <- A + 1;
      }
    }
    X2[n,1..R*C] <- x;
  }  
  //stderr.print("acceptance rate = " + Real(A)/(N*S) + "\n");
  
  /* test distance between the iid and Metropolis samples */
  if !pass(X1, X2) {
    exit(1);
  }
}
