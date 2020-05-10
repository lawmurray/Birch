/**
 * Multivariate inverse-gamma distribution with independent components.
 *
 * This is typically used to establish a conjugate prior for a Bayesian
 * multivariate linear regression with $O$ number of outputs that are
 * conditionally independent given the inputs:
 *
 * $$\begin{align*}
 * \sigma^2_o &\sim \Gamma^{-1}(\alpha_o, \beta_o) \\
 * \boldsymbol{\Sigma} &= \mathrm{diag} \left(\sigma^2_1 \cdots \sigma^2_O \right) \\
 * \mathbf{W} &\sim \mathcal{MN}(\mathbf{M}, \mathbf{A}, \boldsymbol{\Sigma}) \\
 * \mathbf{Y} &\sim \mathcal{N}(\mathbf{X}\mathbf{W}, \boldsymbol{\Sigma}),
 * \end{align*}$$
 *
 * where subscript $o$ denotes the (hyper)parameters of the $o$th element of
 * the output vector, $\mathbf{X}$ are inputs, and $\mathbf{Y}$ are outputs.
 *
 * The relationship is established in code as follows:
 *
 *     σ2:Random<Real[_]>;
 *     α:Real;
 *     β:Real[_];
 *     W:Random<Real[_,_]>;
 *     M:Real[_,_];
 *     U:Real[_,_];
 *     Y:Random<Real[_,_]>;
 *     X:Real[_,_];
 *
 *     σ2 ~ InverseGamma(α, β);
 *     W ~ Gaussian(M, U, σ2);
 *     Y ~ Gaussian(X*W, σ2);
 *
 * The advantage of using this approach over $O$ separate regressions is that
 * expensive covariance operations are shared.
 */
final class IndependentInverseGamma(α:Expression<Real>,
    β:Expression<Real[_]>) < Distribution<Real[_]> {
  /**
   * Shape.
   */
  α:Expression<Real> <- α;
  
  /**
   * Scales.
   */
  β:Expression<Real[_]> <- β;

  function rows() -> Integer {
    return β.rows();
  }

  function simulate() -> Real[_] {
    return transform<Real>(β.value(), @(b:Real) -> Real {
        return simulate_inverse_gamma(α.value(), b); });
  }
  
  function logpdf(x:Real[_]) -> Real {
    return transform_reduce<Real>(x, β.value(), 0.0, @(a:Real, b:Real) -> Real {
        return a + b;
      }, @(x:Real, b:Real) -> Real {
        return logpdf_inverse_gamma(x, α.value(), b);
      });
  }

  function lower() -> Real[_]? {
    return vector(0.0, β.rows());
  }

  function graftIndependentInverseGamma() -> IndependentInverseGamma? {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "IndependentInverseGamma");
    buffer.set("α", α);
    buffer.set("β", β);
  }
}

/**
 * Create inverse-gamma distribution with multiple independent components.
 */
function InverseGamma(α:Expression<Real>, β:Expression<Real[_]>) ->
    IndependentInverseGamma {
  m:IndependentInverseGamma(α, β);
  return m;
}

/**
 * Create inverse-gamma distribution with multiple independent components.
 */
function InverseGamma(α:Expression<Real>, β:Real[_]) ->
    IndependentInverseGamma {
  return InverseGamma(α, Boxed(β));
}

/**
 * Create inverse-gamma distribution with multiple independent components.
 */
function InverseGamma(α:Real, β:Expression<Real[_]>) ->
    IndependentInverseGamma {
  return InverseGamma(Boxed(α), β);
}

/**
 * Create inverse-gamma distribution with multiple independent components.
 */
function InverseGamma(α:Real, β:Real[_]) -> IndependentInverseGamma {
  return InverseGamma(Boxed(α), Boxed(β));
}
