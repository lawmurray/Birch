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

  function start(m:Model) -> Boolean {
    propagations.clear();
    return super.start(m);
  }

  function step() -> Boolean {  
    /* diagnostics */
    ess.pushBack(global.ess(w));
    resample.pushBack(true);
    
    auto f0 <- f;
    auto w0 <- w;
    auto a <- permute_ancestors(ancestors(w0));
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
      do {
        f[n] <- f0[a[n]];
        if (f[n]?) {
          cpp {{
          ++P;
          }}
          (s[n], w[n]) <- f[n]!;
          if (w[n] == -inf) {
            a[n] <- ancestor(w0);
          }
        } else {
          continue <- false;
        }
      } while (continue && w[n] == -inf);
    }
    cpp {{
    P_ = P;
    // can use the normal counter henceforth
    }}
    
    if (continue) {
      /* propagate and weight until one further acceptance, that is discarded
       * for unbiasedness in the normalizing constant estimate */
      f1:(Model, Real)!;
      s1:Model?;
      w1:Real;
      do {
        f1 <- f0[ancestor(w0)];
        if (f1?) {
          P <- P + 1;
          (s1, w1) <- f1!;
        } else {
          continue <- false;
        }
      } while (continue && w1 == -inf);
    }
        
    if (continue) {
      /* update normalizing constant estimate */
      auto W <- log_sum_exp(w);
      w <- w - (W - log(P - 1));
      Z <- Z + W - log(P - 1);
      evidence.pushBack(Z);
      propagations.pushBack(P);
    }
    return continue;
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("propagations", propagations);
  }
}
