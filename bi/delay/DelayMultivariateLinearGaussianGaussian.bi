/*
 * Delayed multivariate linear-Gaussian-Gaussian random variate.
 */
class DelayMultivariateLinearGaussianGaussian(x:Random<Real[_]>&,
    A:Real[_,_], m:DelayMultivariateGaussian, c:Real[_], S:Real[_,_]) <
    DelayMultivariateGaussian(x, A*m.μ + c, A*cholinv(m.Λ)*trans(A) + S) {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;
    
  /**
   * Mean.
   */
  m:DelayMultivariateGaussian& <- m;

  /**
   * Offset.
   */
  c:Real[_] <- c;
  
  /**
   * Likelihood covariance.
   */
  L:Real[_,_] <- cholinv(S);

  function update(x:Real[_]) {
    (m!.μ, m!.Λ) <- update_multivariate_linear_gaussian_gaussian(x, A, m!.μ,
        m!.Λ, c, L);
  }
}

function DelayMultivariateLinearGaussianGaussian(x:Random<Real[_]>&,
    A:Real[_,_], μ:DelayMultivariateGaussian, c:Real[_], Σ:Real[_,_]) ->
    DelayMultivariateLinearGaussianGaussian {
  m:DelayMultivariateLinearGaussianGaussian(x, A, μ, c, Σ);
  μ.setChild(m);
  return m;
}
