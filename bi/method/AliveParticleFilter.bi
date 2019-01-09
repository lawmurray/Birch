/**
 * Alive particle filter. When propagating and weighting particles, the
 * alive particle filter maintains $N$ particles with non-zero weight, rather
 * than $N$ particles in total as with the standard particle filter.
 */
class AliveParticleFilter < ParticleFilter {
  /**
   * For each checkpoint, the number of propagations that were performed to
   * achieve $N$ acceptances.
   */
  propagations:List<Integer>;

  function initialize(m:Model) {
    super.initialize(m);
    propagations.clear();
    trigger <- 1.0;  // always resample
  }

  function propagate() -> Boolean {  
    /* diagnostics */    
    auto x0 <- x;
    auto w0 <- w;
    auto P <- 0;  // number of propagations
    
    /* propagate and weight until nparticles acceptances; the first
     * nparticles proposals are drawn using the standard resampler, then each
     * is propagated until it has non-zero weight, proposal alternatives with
     * a categorical draw */  
    auto continue <- true;
    cpp {{
    /* use an atomic to accumulate the number of propagations within the
     * parallel loop */
    std::atomic<int> P;
    P = 0;
    }}
    parallel for (n:Integer in 1..nparticles) {
      v:Real?;
      do {
        x[n] <- clone<Model>(x0[a[n]]);
        v <- x[n].step();
        if v? {
          cpp {{
          ++P;
          }}
          if (v! == -inf) {
            a[n] <- ancestor(w0);
          } else {
            w[n] <- v!;
          }
        } else {
          continue <- false;
        }
      } while (continue && v! == -inf);
    }
    cpp {{
    P_ = P;
    // can use the normal counter henceforth
    }}
    
    if (continue) {
      /* propagate and weight until one further acceptance, that is discarded
       * for unbiasedness in the normalizing constant estimate */
      v:Real?;
      do {
        auto x1 <- clone<Model>(x0[ancestor(w0)]);
        v <- x1.step();
        if v? {
          P <- P + 1;
        } else {
          continue <- false;
        }
      } while (continue && v! == -inf);
    }
    propagations.pushBack(P);
    return continue;
  }
  
  function reduce() {
    /* effective sample size */
    ess.pushBack(global.ess(w));
    if (!(ess.back() > 0.0)) {  // may be nan
      error("particle filter degenerated.");
    }
  
    /* normalizing constant estimate */
    auto W <- log_sum_exp(w);
    auto P <- propagations.back();
    w <- w - (W - log(P - 1));
    Z <- Z + W - log(P - 1);
    evidence.pushBack(Z);
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("propagations", propagations);
  }
}
