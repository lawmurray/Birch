/*
 * Test a continuous distribution.
 *
 * - q: The distribution. 
 * - N: Number of partitions for Riemann (midpoint) estimate.
 */
function test_cdf(q:Distribution<Real>, N:Integer) {
  auto from <- q.lower();
  if !from? {
    from <- q.quantile(1.0e-6);
    assert from?;
  }
  auto to <- q.upper();
  if !to? {
    to <- q.quantile(1.0 - 1.0e-6);
    assert to?;
  }

  auto P <- 0.5/N;
  for auto n in 1..N {
    auto x <- from! + (n - 0.5)*(to! - from!)/N;
    auto C <- q.cdf(x)!;
    P <- P + q.pdf(x)*(to! - from!)/N;
   
    auto δ <- abs(C - P);
    auto ε <- 4.0/n;
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
  auto from <- q.lower();
  if !from? {
    from <- q.quantile(1.0e-6);
    assert from?;
  }
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
    auto ε <- 4.0/(x - from! + 1);
    auto failed <- δ > ε;
    if failed {
      stderr.print("failed on value " + x + ", " + δ + " > " + ε + "\n");
      exit(1);
    }
  }
}
