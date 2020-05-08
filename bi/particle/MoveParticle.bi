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
      π <- π + w;
      this.z.pushBack(z!);
    } else {
      this.z.pushBack(Boxed(0.0));
    }
    return w;
  }
  
  /**
   * Bring the prior up-to-date after one or more calls to `add()`.
   */
  function prior() {
    for t in (p.size() + 1)..z.size() {
      auto z <- this.z.get(t);
      auto p <- z.prior();
      if p? {
        π <- π + p!.pilot();
        this.p.pushBack(p!);
      } else {
        this.p.pushBack(Boxed(0.0));
      }
    }
    assert z.size() == p.size();
  }
  
  /**
   * Bring gradients up-to-date after one or more calls to `add()` then
   * `prior()`.
   */
  function grad() {
    while n + 1 < z.size() {
      n <- n + 1;
      z.get(n).grad(1.0);
      p.get(n).grad(1.0);
    }
  } 
  
  /**
   * Move the particle.
   */
  function move() {
    π <- 0.0;
    n <- 0;
    for t in 1..z.size() {
      π <- π + z.get(t).move();
      π <- π + p.get(t).move();
    }
    grad();
  }
  
  /**
   * Remove the first step. Latent random variates for the step are realized
   * and explicitly escaped, in order that they are no longer eligible for
   * moving. This is used by resample-move particle filters with a finite
   * lag to make steps outside of that lag ineligible for move.
   */
  function popFront() {
  
  }
}

/**
 * Create a MoveParticle.
 */
function MoveParticle(m:Model) -> MoveParticle {
  o:MoveParticle(m);
  return o;
}
