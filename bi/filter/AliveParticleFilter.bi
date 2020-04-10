/**
 * Alive particle filter. When propagating and weighting particles, the
 * alive particle filter maintains $N$ particles with non-zero weight, rather
 * than $N$ particles in total as with the standard particle filter.
 *
 * The ParticleFilter class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Filter.svg"></object>
 * </center>
 */
class AliveParticleFilter < ParticleFilter {
  /**
   * Number of propagations for each particle, along with the extra particle
   * to be discarded.
   */
  p:Integer[_];

  override function filter(archetype:Model, t:Integer) {
    auto x0 <- x;
    auto w0 <- w;
    p <- vector(0, nparticles + 1);
    parallel for n in 1..nparticles + 1 {
      if n <= nparticles {
        x[n] <- clone(x0[a[n]]);
        w[n] <- play.handle(x[n].simulate(t));
        p[n] <- 1;
        while w[n] == -inf {  // repeat until weight is positive
          a[n] <- global.ancestor(w0);
          x[n] <- clone(x0[a[n]]);
          p[n] <- p[n] + 1;
          w[n] <- play.handle(x[n].simulate(t));
        }
      } else {
        /* propagate and weight until one further acceptance, which is
         * discarded for unbiasedness in the normalizing constant
         * estimate */
        auto w' <- 0.0;
        p[n] <- 0;
        do {
          auto a' <- global.ancestor(w0);
          auto x' <- clone(x0[a']);
          p[n] <- p[n] + 1;
          w' <- play.handle(x'.simulate(t));
        } while w' == -inf;  // repeat until weight is positive
      }
    }    
    reduce();
  }
  
  override function reduce() {
    npropagations <- sum(p);
    super.reduce();
  }
}
