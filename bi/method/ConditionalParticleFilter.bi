/**
 * Conditional particle filter. Conditions on a previously known path
 * $x'_{1:T}$.
 */
class ConditionalParticleFilter < ParticleFilter {
  /**
   * Conditioned path, drawn from previous iteration.
   */
  fN:List<(Model,Real)!>;
   
  /**
   * All paths, in current iteration.
   */
  F:List<(Model,Real)!>[_];
   
  function start(m:Model) {
    super.start(m);
    F1:List<(Model,Real)!>[nparticles];
    F <- F1;
  }

  /**
   * Resample particles.
   */
  function resample() {
    if fN.empty() {
      a <- multinomial_ancestors(w);
    } else {
      a <- multinomial_conditional_ancestors(w);
    }
    w <- vector(0.0, nparticles);
  }
  
  function copy() {
    super.copy();
    auto F1 <- F;
    parallel for auto n in 1..nparticles {
      F[n] <- clone<List<(Model,Real)!>>(F1[a[n]]);
    }
  }

  function propagate() -> Boolean {
    auto cont <- true;
    auto N <- nparticles;
    
    if !fN.empty() {
      /* condition on the given path */
      w1:Real;
      (x[N], w1) <- fN.front()!;
      F[N].pushBack(fN.front());
      w[N] <- w[N] + w1;
      N <- N - 1;
      fN.popFront();
    }
    parallel for auto n in 1..N {
      w1:Real;
      if f[n]? {
        (x[n], w1) <- f[n]!;
        F[n].pushBack(clone<(Model,Real)!>(f[n]));
        w[n] <- w[n] + w1;
      } else {
        cont <- false;      
      }
    }
    return cont;
  }

  function finish() {
    super.finish();
    fN <- F[b];
  }
}
