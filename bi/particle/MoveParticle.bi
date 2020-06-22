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
  function likelihood(z:Expression<Real>?) -> Real {
    auto z' <- z;
    if !z'? {
      z' <- box(0.0);
    }
    auto w <- z'!.pilot();
    zs.pushBack(z'!);
    π <- π + w;
    return w;
  }

  /**
   * Catch up priors after one or more calls to `likelihood()`.
   */
  function prior() {
    assert vs.size() == ps.size();
    for i in (ps.size() + 1)..zs.size() {
      vs.pushBack();
      auto z <- zs.get(i);
      auto p <- z.prior(vs);
      if !p? {
        p <- box(0.0);
      }
      π <- π + p!.pilot();
      ps.pushBack(p!);
    }
    assert ps.size() == zs.size();
    assert vs.size() == zs.size();
  }
  
  /**
   * Catch up gradients after one or more calls to `prior()`.
   */
  function grad() {
    assert ps.size() == zs.size();
    assert vs.size() == zs.size();
    while n < zs.size() {
      n <- n + 1;
      zs.get(n).grad(1.0);
      ps.get(n).grad(1.0);
    }
    assert n == zs.size();
    assert n == ps.size();
    assert n == vs.size();
  }

  /**
   * Remove the oldest step after a call to `grad()`.
   *
   * This must be called after `grad()` and not at any other time, as at this
   * point [Expression](../classes/Expression) evaluation counts are at zero.
   * Calling it at any time with nonzero counts expression states invalidate
   * the state of those expressions.
   */
  function truncate() {
    /* make variables from the oldest time step constant, so that they are
     * no longer eligible to move */
    for j in 1..vs.size(1) {
      vs.get(1, j).makeConstant();
    }
  
    /* update state */
    π <- π - zs.front().get() - ps.front().get();
    n <- n - 1;
    zs.popFront();
    ps.popFront();
    vs.popFront();
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
  o:MoveParticle(m);
  return o;
}
