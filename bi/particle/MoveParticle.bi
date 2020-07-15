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
   * Number of steps.
   */
  function size() -> Integer {
    return zs.size();
  } 

  /**
   * Add a new step.
   *
   * - t: The step number, beginning at 0.
   * - z: Expression giving the incremental log-likelihood for th new step.
   *
   * Returns: Incremental log-likelihood; zero if the argument is `nil`.
   */
  function augment(t:Integer, z:Expression<Real>?) -> Real {
    /* likelihood */
    auto z' <- z;
    if !z'? {
      z' <- box(0.0);
    }
    auto w <- z'!.pilot(t);
    π <- π + w;
    zs.pushBack(z'!);
    
    /* prior */
    vs.pushBack();
    auto p <- z'!.prior(vs);
    if !p? {
      p <- box(0.0);
    }
    π <- π + p!.pilot(t);
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
     * ineligible for moves in future; that's what we want */
    if !zs.empty() {
      π <- π - zs.front().get();
      zs.popFront();
    }
    if !ps.empty() {
      π <- π - ps.front().get();
      ps.popFront();
    }
    if !vs.empty() {
      vs.popFront();
    }

    assert ps.size() == zs.size();
    assert vs.size() == zs.size();
  }
  
  /**
   * Compute gradient.
   */
  function grad(gen:Integer) {
    assert ps.size() == zs.size();
    auto L <- zs.size();    
    for l in 1..L {
      zs.get(l).grad(gen, 1.0);
      ps.get(l).grad(gen, 1.0);
    }
  }

  /**
   * Move.
   *
   * - κ: Markov kernel.
   */
  function move(gen:Integer, κ:Kernel) {
    assert ps.size() == zs.size();
    auto L <- zs.size();    
    π <- 0.0;
    for l in 1..L {
      π <- π + zs.get(l).move(gen, κ);
      π <- π + ps.get(l).move(gen, κ);
    }
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
