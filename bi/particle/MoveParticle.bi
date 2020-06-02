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
  z:Vector<Expression<Real>>;
  
  /**
   * Log-priors. Each element is a lazy expression giving a single step's
   * contribution to the log-prior.
   */
  p:Vector<Expression<Real>>;
  
  /**
   * Variables. Each row collects the variables encountered in a single step.
   */
  vars:RaggedArray<DelayExpression>;
  
  /**
   * Log-posterior density.
   */
  π:Real <- 0.0;
  
  /**
   * Step up to which gradients have been evaluated.
   */
  n:Integer <- 0;
  
  /**
   * Add the deferred log-likelihood for a new step.
   *
   * - z: Log-likelihood.
   *
   * Returns: The evaluation of the expression.
   */
  function add(z:Expression<Real>?) -> Real {
    auto w <- 0.0;
    if z? {
      w <- z!.pilot();
      this.z.pushBack(z!);
    } else {
      auto z <- box(0.0);
      w <- z.pilot();  // must evaluate for later grad()
      this.z.pushBack(z);
    }
    π <- π + w;
    return w;
  }
  
  /**
   * Update the prior after one or more calls to `add()`.
   */
  function prior() {
    assert vars.size() == p.size();
    for t in (p.size() + 1)..z.size() {
      vars.pushBack();
      auto z <- this.z.get(t);
      auto p <- z.prior(vars);
      if p? {
        π <- π + p!.pilot();
        this.p.pushBack(p!);
      } else {
        auto p <- box(0.0);
        π <- π + p.pilot();  // must evaluate for later grad()
        this.p.pushBack(p);
      }
    }
    assert p.size() == z.size();
    assert vars.size() == z.size();
  }
  
  /**
   * Bring gradients up-to-date after one or more calls to `add()` then
   * `prior()`.
   */
  function grad() {
    assert p.size() == z.size();
    while n < z.size() {
      n <- n + 1;
      z.get(n).grad(1.0);
      p.get(n).grad(1.0);
    }
    assert n == z.size();
    assert n == p.size();
  }

  /**
   * Move the particle.
   *
   * - κ: Markov kernel.
   */
  function move(κ:Kernel) {
    assert n == z.size();
    assert n == p.size();
    π <- 0.0;
    for t in 1..n {
      π <- π + z.get(t).move(κ);
      π <- π + p.get(t).move(κ);
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
    assert vars.size() == x'.vars.size();
    auto q <- 0.0;
    for t in 1..vars.size() {
      assert vars.size(t) == x'.vars.size(t);
      for i in 1..vars.size(t) {
        q <- q + vars.get(t, i).logpdf(x'.vars.get(t, i), κ);
      }
    }
    return q;
  }
  
  /**
   * Remove the first step. Latent random variates for the step are realized
   * and explicitly escaped, in order that they are no longer eligible for
   * moving. This is used by resample-move particle filters with a finite
   * lag to make steps outside of that lag ineligible for move.
   */
  //function popFront() {
  //
  //}
}

/**
 * Create a MoveParticle.
 */
function MoveParticle(m:Model) -> MoveParticle {
  o:MoveParticle(m);
  return o;
}
