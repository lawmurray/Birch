/*
 * Test a continuous distribution.
 *
 * - q: The distribution. 
 * - N: Number of partitions for Riemann (midpoint) estimate.
 */
function test_cdf(q:Distribution<Real>, N:Integer) {
  /* lower bound on test interval */
  auto from <- q.lower();
  if !from? {
    from <- q.quantile(1.0e-6);
    assert from?;
  }
  
  /* upper bound on test interval */
  auto to <- q.upper();
  if !to? {
    to <- q.quantile(1.0 - 1.0e-6);
    if !to? {
      /* crudely search for an upper bound */
      auto u <- 1.0;
      while q.pdf(u) > 1.0e-6 {
        u <- 2.0*u;
      }
      to <- u;
    }
    assert to?;
  }

  auto P <- 0.5/N;
  for auto n in 1..N {
    auto x <- from! + (n - 0.5)*(to! - from!)/N;
    auto C <- q.cdf(x)!;
    P <- P + q.pdf(x)*(to! - from!)/N;
   
    auto δ <- abs(C - P);
    auto ε <- 10.0/n;
    auto failed <- δ > ε;
    if failed {
      stderr.print("failed on step " + n + ", " + δ + " > " + ε + "\n");
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
  /* lower bound on test interval */
  auto from <- q.lower();
  if !from? {
    from <- q.quantile(1.0e-6);
    assert from?;
  }

  /* upper bound on test interval */
  auto to <- q.upper();
  if !to? {
    to <- q.quantile(1.0 - 1.0e-6);
    assert to?;
  }
  
  auto P <- 0.0;
  for auto x in from!..to! {
    auto C <- q.cdf(x)!;
    P <- P + q.pdf(x);
    
    auto δ <- abs(C - P);
    auto ε <- 10.0/(x - from! + 1);
    auto failed <- δ > ε;
    if failed {
      stderr.print("failed on value " + x + ", " + δ + " > " + ε + "\n");
      exit(1);
    }
  }
}
