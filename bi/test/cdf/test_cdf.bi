/*
 * Test a continuous distribution.
 *
 * - q: The distribution. 
 * - N: Number of partitions for Riemann (midpoint) estimate.
 */
function test_cdf(q:Distribution<Real>, N:Integer) {
  auto failed <- false;
  
  /* lower bound on test interval */
  auto from <- q.lower();
  if from? {
    /* test the lower bound against the quantile */
    auto test <- q.quantile(0.0);
    if test? && abs(from! - test!) > 1.0/N {
      failed <- true;
      stderr.print("lower bound and quantile comparison failed\n");
    }
  } else {
    from <- q.quantile(1.0/N);
    assert from?;
  }

  /* upper bound on test interval */
  auto to <- q.upper();
  if to? {
    /* test the upper bound against the quantile */
    auto test <- q.quantile(1.0);
    if test? && abs(to! - test!) > 1.0/N {
      failed <- true;
      stderr.print("upper bound and quantile comparison failed\n");
    }
  } else {
    to <- q.quantile(1.0 - 1.0/N);
    if !to? {
      /* search for a rough upper bound for the interval */
      auto u <- 1.0;
      while q.pdf(u) > 1.0/N {
        u <- 2.0*u;
      }
      to <- u;
    }
    assert to?;
  }

  /* compare sum of pdf to cdf evaluations */
  auto P <- 0.5/N;
  for auto n in 1..N {
    auto x <- from! + (n - 0.5)*(to! - from!)/N;
    auto C <- q.cdf(x)!;
    P <- P + q.pdf(x)*(to! - from!)/N;
   
    auto δ <- abs(C - P);
    auto ε <- 10.0/sqrt(N);
    if δ > ε {
      failed <- true;
      stderr.print("failed on step " + n + ", " + δ + " > " + ε + "\n");
    }
    if failed {
      exit(1);
    }
  }
}

/*
 * Test a discrete distribution.
 *
 * - q: The distribution.
 */
function test_cdf(q:Distribution<Integer>) {
  auto failed <- false;
  
  /* lower bound on test interval */
  auto from <- q.lower();
  if from? {
    /* test the lower bound against the quantile */
    auto test <- q.quantile(0.0);
    if test? && abs(from! - test!) > 1.0e-6 {
      failed <- true;
      stderr.print("lower bound and quantile comparison failed\n");
    }
  } else {
    from <- q.quantile(1.0e-6);
    assert from?;
  }

  /* upper bound on test interval */
  auto to <- q.upper();
  if to? {
    /* test the upper bound against the quantile */
    auto test <- q.quantile(1.0);
    if test? && abs(to! - test!) > 1.0e-6 {
      failed <- true;
      stderr.print("upper bound and quantile comparison failed\n");
    }
  } else {
    to <- q.quantile(1.0 - 1.0e-6);
    assert to?;
  }
  
  /* compare sum of pdf to cdf evaluations */
  auto P <- 0.0;
  for auto x in from!..to! {
    auto C <- q.cdf(x)!;
    P <- P + q.pdf(x);
    
    auto δ <- abs(C - P);
    auto ε <- 10.0/sqrt(x - from! + 1);
    if δ > ε {
      failed <- true;
      stderr.print("failed on value " + x + ", " + δ + " > " + ε + "\n");
    }
    if failed {
      exit(1);
    }
  }
}
