/**
 * Particle for use with MoveParticleFilter.
 *
 * - m: Model.
 *
 * The Particle class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Particle.svg"></object>
 * </center>
 */
class MoveParticle(m:Model) < Particle(m) {
  /**
   * Log-likelihoods. Each element is a lazy expression giving a single
   * step's contribution to the log-likelihood.
   */
  zs:Vector<Expression<Real>>;
  
  /**
   * Log-priors. Each element is a lazy expression giving a single step's
   * contribution to the log-prior.
   */
  ps:Vector<Expression<Real>>;
  
  /**
   * Variables. Each row collects the variables encountered in a single step.
   */
  vs:RaggedArray<DelayExpression>;
  
  /**
   * Log-posterior density.
   */
  π:Real <- 0.0;
  
  /**
   * Step up to which gradients have been evaluated.
   */
  n:Integer <- 0;

  /**
   * Number of steps.
   */
  function size() -> Integer {
    return zs.size();
  } 

  /**
   * Add a new step.
   *
   * - z: Expression giving the incremental log-likelihood for th new step.
   *
   * Returns: Incremental log-likelihood; zero if the argument is `nil`.
   */
  function augment(z:Expression<Real>?) -> Real {
    /* likelihood */
    auto z' <- z;
    if !z'? {
      z' <- box(0.0);
    }
    auto w <- z'!.pilot();
    π <- π + w;
    zs.pushBack(z'!);
    
    /* prior */
    vs.pushBack();
    auto p <- z'!.prior(vs);
    if !p? {
      p <- box(0.0);
    }
    π <- π + p!.pilot();
    ps.pushBack(p!);

    assert ps.size() == zs.size();
    assert vs.size() == zs.size();
    
    return w;
  }

  /**
   * Remove the oldest step.
   */
  function truncate() {
    /* calling value() on these expressions has the side effect of making
     * them constant, so that Random objects appearing in them will be
     * ineligible for moves in future */
    if !zs.empty() {
      π <- π - zs.front().value();
      zs.popFront();
    }
    if !ps.empty() {
      π <- π - ps.front().value();
      ps.popFront();
    }
    if !vs.empty() {
      vs.popFront();
    }
    if n > 0 {
      n <- n - 1;
    }
  }
  
  /**
   * Catch up gradient computations after one or more calls to `augment()`.
   */
  function grad() {
    assert ps.size() == zs.size();
    assert vs.size() == zs.size();
    while n < zs.size() {
    assert n >= 0;
      n <- n + 1;
      zs.get(n).grad(1.0);
      ps.get(n).grad(1.0);
    }
    assert n == zs.size();
    assert n == ps.size();
    assert n == vs.size();
  }

  /**
   * Move the particle after one or more calls to `grad()`.
   *
   * - κ: Markov kernel.
   */
  function move(κ:Kernel) {
    assert n == zs.size();
    assert n == ps.size();
    assert n == vs.size();
    π <- 0.0;
    for i in 1..n {
      π <- π + zs.get(i).move(κ);
      π <- π + ps.get(i).move(κ);
    }
    n <- 0;
  }

  /**
   * Compute the log-pdf of a proposed state. This object is considered the
   * current state, $x$.
   *
   * - x': Proposed state $x^\prime$.
   * - κ: Markov kernel.
   *
   * Returns: $\log q(x^\prime \mid x)$.
   */
  function logpdf(x':MoveParticle, κ:Kernel) -> Real {
    assert size() == x'.size();
    assert vs.size() == x'.vs.size();
    auto q <- 0.0;
    for i in 1..vs.size() {
      assert vs.size(i) == x'.vs.size(i);
      for j in 1..vs.size(i) {
        q <- q + vs.get(i,j).logpdf(x'.vs.get(i,j), κ);
      }
    }
    return q;
  }
}

/**
 * Create a MoveParticle.
 */
function MoveParticle(m:Model) -> MoveParticle {
  return construct<MoveParticle>(m);
}
