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
final class MultivariateNormalInverseGamma(μ:Expression<Real[_]>,
    Σ:Expression<Real[_,_]>, σ2:InverseGamma) < Distribution<Real[_]> {
  /**
   * Precision.
   */
  Λ:Expression<LLT> <- llt(inv(llt(Σ)));

  /**
   * Precision times mean.
   */
  ν:Expression<Real[_]> <- matrix(Λ)*μ;

  /**
   * Variance shape.
   */
  α:Expression<Real> <- σ2.α;

  /**
   * Variance scale accumulator.
   */
  γ:Expression<Real> <- σ2.β + 0.5*dot(μ, ν);

  /**
   * Variance scale.
   */
  σ2:InverseGamma& <- σ2;

  function rows() -> Integer {
    return ν.rows();
  }

  function simulate() -> Real[_] {
    return simulate_multivariate_normal_inverse_gamma(ν.value(), Λ.value(),
        α.value(), gamma_to_beta(γ.value(), ν.value(), Λ.value()));
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_normal_inverse_gamma(x, ν.value(), Λ.value(),
        α.value(), gamma_to_beta(γ.value(), ν.value(), Λ.value()));
  }

  function update(x:Real[_]) {
    (σ2.α, σ2.β) <- box(update_multivariate_normal_inverse_gamma(x, ν.value(),
        Λ.value(), α.value(), gamma_to_beta(γ.value(), ν.value(), Λ.value())));
  }

  function downdate(x:Real[_]) {
    (σ2.α, σ2.β) <- box(downdate_multivariate_normal_inverse_gamma(x, ν.value(),
        Λ.value(), α.value(), gamma_to_beta(γ.value(), ν.value(), Λ.value())));
  }

  function graftMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
      MultivariateNormalInverseGamma? {
    prune();
    if σ2 == compare {
      return this;
    } else {
      return nil;
    }
  }

  function link() {
    σ2.setChild(this);
  }
  
  function unlink() {
    σ2.releaseChild(this);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MultivariateNormalInverseGamma");
    buffer.set("μ", solve(Λ.value(), ν.value()));
    buffer.set("Σ", inv(Λ.value()));
    buffer.set("α", α.value());
    buffer.set("β", gamma_to_beta(γ.value(), ν.value(), Λ.value()));
  }
}

function MultivariateNormalInverseGamma(μ:Expression<Real[_]>,
    Σ:Expression<Real[_,_]>, σ2:InverseGamma) ->
    MultivariateNormalInverseGamma {
  m:MultivariateNormalInverseGamma(μ, Σ, σ2);
  m.link();
  return m;
}

/*
 * Compute the variance scale from the variance scale accumulator and other
 * parameters.
 */
function gamma_to_beta(γ:Real, ν:Real[_], Λ:LLT) -> Real {
  return γ - 0.5*dot(solve(cholesky(Λ), ν));
}
