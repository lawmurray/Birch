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

  function start(m:Model) {
    super.start(m);
    propagations.clear();
  }

  function step() -> Boolean {  
    /* diagnostics */
    ess.pushBack(global.ess(w));
    resample.pushBack(true);
    
    auto f0 <- f;
    auto w0 <- w;
    auto a <- permute_ancestors(ancestors(w0));
    auto P <- 0;  // number of proposals

    /* propagate and weight until nparticles acceptances; the first
     * nparticles proposals are drawn using the standard resampler, then each
     * is propagated until it has non-zero weight, proposal alternatives with
     * a categorical draw */  
    auto continue <- true;
    parallel for (n:Integer in 1..nparticles) {
      f[n] <- f0[a[n]];
      if (f[n]?) {
        P <- P + 1;
      } else {
        continue <- false;
      }
      while (continue && f[n]!.w == -inf) {
        f[n] <- f0[ancestor(w0)];
        if (f[n]?) {
          P <- P + 1;
        } else {
          continue <- false;
        }
      }
      w[n] <- f[n]!.w;
    }
    
    if (continue) {
      /* propagate and weight until one further acceptance, that is discarded
       * for unbiasedness in the normalizing constant estimate */
      auto f1 <- f0[ancestor(w0)];
      if (f1?) {
        P <- P + 1;
      } else {
        continue <- false;
      }
      while (continue && f1!.w == -inf) {
        f1 <- f0[ancestor(w0)];
        if (f1?) {
          P <- P + 1;
        } else {
          continue <- false;
        }
      }
    }
    
    if (continue) {
      /* update normalizing constant estimate */
      auto W <- log_sum_exp(w);
      w <- w - (W - log(nparticles));
      if (evidence.empty()) {
        evidence.pushBack(W - log(nparticles));
      } else {
        evidence.pushBack(evidence.back() + W - log(nparticles));
      }
      propagations.pushBack(P);
    }
    
    return continue;
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("propagations", propagations);
  }
}
