/**
 * Multivariate normal-inverse-gamma distribution.
 *
 * This represents the joint distribution:
 *
 * $$\sigma^2 \sim \mathrm{Inverse-Gamma}(\alpha, \beta)$$
 * $$x \mid \sigma^2 \sim \mathrm{N}(\mu, Σ\sigma^2),$$
 *
 * which may be denoted:
 *
 * $$(x, \sigma^2) \sim \mathrm{Normal-Inverse-Gamma(\mu, Σ, \alpha, \beta),$$
 *
 * and is a conjugate prior of a Gaussian distribution with both unknown mean
 * and variance. The variance scaling is independent and identical in the
 * sense that all components of $x$ share the same $\sigma^2$.
 *
 * In model code, it is not usual to use this class directly. Instead,
 * establish a conjugate relationship via code such as the following:
 *
 *     σ2 ~ InverseGamma(α, β);
 *     x ~ Gaussian(μ, Σ*σ2);
 *     y ~ Gaussian(x, σ2);
 *
 * where the last argument in the distribution of `y` must appear in the
 * last argument of the distribution of `x`. The operation of `Σ` on `σ2` may
 * be multiplication on the left (as above) or the right, or division on the
 * right.
 */
final class MultivariateNormalInverseGamma(μ:Real[_], Σ:Real[_,_],
    σ2:InverseGamma) < Distribution<Real[_]> {
  /**
   * Precision.
   */
  Λ:LLT <- llt(inv(llt(Σ)));

  /**
   * Precision times mean.
   */
  ν:Real[_] <- Λ*μ;

  /**
   * Variance shape.
   */
  α:Real <- σ2.α;

  /**
   * Variance scale accumulator.
   */
  γ:Real <- σ2.β + 0.5*dot(μ, ν);

  /**
   * Variance scale.
   */
  σ2:InverseGamma& <- σ2;

  function rows() -> Integer {
    return length(ν);
  }

  function simulate() -> Real[_] {
    return simulate_multivariate_normal_inverse_gamma(ν, Λ, α,
        gamma_to_beta(γ, ν, Λ));
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_normal_inverse_gamma(x, ν, Λ, α,
        gamma_to_beta(γ, ν, Λ));
  }

  function update(x:Real[_]) {
    (σ2.α, σ2.β) <- update_multivariate_normal_inverse_gamma(x, ν, Λ, α,
        gamma_to_beta(γ, ν, Λ));
  }

  function downdate(x:Real[_]) {
    (σ2.α, σ2.β) <- downdate_multivariate_normal_inverse_gamma(x, ν, Λ, α,
        gamma_to_beta(γ, ν, Λ));
  }

  function graft() -> Distribution<Real[_]> {
    prune();
    return this;
  }

  function graftMultivariateNormalInverseGamma() ->
      MultivariateNormalInverseGamma? {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MultivariateNormalInverseGamma");
    buffer.set("μ", solve(Λ, ν));
    buffer.set("Σ", inv(Λ));
    buffer.set("α", α);
    buffer.set("β", gamma_to_beta(γ, ν, Λ));
  }
}

function MultivariateNormalInverseGamma(μ:Real[_], Σ:Real[_,_],
    σ2:InverseGamma) -> MultivariateNormalInverseGamma {
  m:MultivariateNormalInverseGamma(μ, Σ, σ2);
  σ2.setChild(m);
  return m;
}

/*
 * Compute the variance scale from the variance scale accumulator and other
 * parameters.
 */
function gamma_to_beta(γ:Real, ν:Real[_], Λ:LLT) -> Real {
  return γ - 0.5*dot(solve(cholesky(Λ), ν));
}
