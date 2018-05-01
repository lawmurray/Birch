/**
 * Test Gaussian-Gaussian conjugacy.
 */
program test_gaussian_gaussian(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
  μ_0:Real <- simulate_uniform(-10.0, 10.0);
  σ2_0:Real <- simulate_uniform(0.0, 10.0);
  σ2_1:Real <- simulate_uniform(0.0, 10.0);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestGaussianGaussian(μ_0, σ2_0, σ2_1);
    m.initialize();
    X1[i,1..2] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestGaussianGaussian(μ_0, σ2_0, σ2_1);
    m.initialize();
    X2[i,1..2] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}
