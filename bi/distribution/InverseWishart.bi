/**
 * Inverse Wishart distribution.
 *
 * This is typically used to establish a conjugate prior for a Bayesian
 * multivariate linear regression:
 *
 * $$\begin{align*}
 * \mathbf{\Sigma} &\sim \mathcal{W}^{-1}(\mathbf{\Psi}, \nu) \\
 * \mathbf{W} &\sim \mathcal{MN}(\mathbf{M}, \mathbf{A}, \boldsymbol{\Sigma}) \\
 * \mathbf{y}_n &\sim \mathcal{N}(\mathbf{W}\mathbf{x}_n, \boldsymbol{\Sigma}),
 * \end{align*}$$
 *
 * where $\mathbf{x}_n$ is the $n$th input and $\mathbf{y}_n$ the $n$th
 * output.
 *
 * The relationship is established in code as follows:
 *
 *     Ψ:Real[_,_];
 *     ν:Real;
 *     Σ:Random<Real[_,_]>;
 *     W:Random<Real[_,_]>;
 *     M:Real[_,_];
 *     A:Real[_,_];
 *     y:Random<Real[_]>;
 *     x:Real[_];
 *
 *     Σ ~ InverseWishart(Ψ, ν);
 *     W ~ Gaussian(M, A, Σ);
 *     y ~ Gaussian(W*x, Σ);
 */
final class InverseWishart(Ψ:Expression<Real[_,_]>, ν:Expression<Real>) <
    Distribution<Real[_,_]> {
  /**
   * Scale.
   */
  Ψ:Expression<Real[_,_]> <- Ψ;
  
  /**
   * Degrees of freedom.
   */
  ν:Expression<Real> <- ν;

  function valueForward() -> Real[_,_] {
    assert !delay?;
    return simulate_inverse_wishart(Ψ, ν);
  }

  function observeForward(X:Real[_,_]) -> Real {
    assert !delay?;
    return logpdf_inverse_wishart(X, Ψ, ν);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayInverseWishart(future, futureUpdate, Ψ, ν);
    }
  }

  function graftInverseWishart() -> DelayInverseWishart? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayInverseWishart(future, futureUpdate, Ψ, ν);
    }
    return DelayInverseWishart?(delay);
  }
}

/**
 * Create inverse-Wishart distribution.
 */
function InverseWishart(Ψ:Expression<Real[_,_]>, ν:Expression<Real>) ->
    InverseWishart {
  m:InverseWishart(Ψ, ν);
  return m;
}

/**
 * Create inverse-Wishart distribution.
 */
function InverseWishart(Ψ:Expression<Real[_,_]>, ν:Real) -> InverseWishart {
  return InverseWishart(Ψ, Boxed(ν));
}

/**
 * Create inverse-Wishart distribution.
 */
function InverseWishart(Ψ:Real[_,_], ν:Expression<Real>) -> InverseWishart {
  return InverseWishart(Boxed(Ψ), ν);
}

/**
 * Create inverse-Wishart distribution.
 */
function InverseWishart(Ψ:Real[_,_], ν:Real) -> InverseWishart {
  return InverseWishart(Boxed(Ψ), Boxed(ν));
}
