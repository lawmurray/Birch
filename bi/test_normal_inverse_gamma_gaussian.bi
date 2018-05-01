/**
 * Test inverse-gamma-Gaussian conjugacy.
 */
program test_normal_inverse_gamma_gaussian(N:Integer <- 10000) {
  X1:Real[N,3];
  X2:Real[N,3];
  
  μ:Real <- simulate_uniform(-10.0, 10.0);
  a2:Real <- simulate_uniform(0.0, 10.0);
  α:Real <- simulate_uniform(0.0, 10.0);
  β:Real <- simulate_uniform(0.0, 10.0);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestNormalInverseGammaGaussian(μ, a2, α, β);
    m.initialize();
    X1[i,1..3] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestNormalInverseGammaGaussian(μ, a2, α, β);
    m.initialize();
    X2[i,1..3] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}
