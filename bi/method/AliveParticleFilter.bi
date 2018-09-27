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
  propagations:Integer[_];

  function start() {
    super.start();
    propagations <- vector(0, T);
  }

  function step(t:Integer) {  
    /* diagnostics */
    e[t] <- ess(w);
    r[t] <- true;
    
    f0:Model![_] <- f;
    w0:Real[_] <- w;
    a:Integer[_] <- ancestors(w0);
    P:Integer;  // number of proposals
    cpp {{
    std::atomic<int> P;
    P = 0;
    }}

    /* propagate and weight until N acceptances; the first N proposals are
     * drawn using the standard (stratified) resampler, then each is
     * is propagated until it has non-zero weight, proposal alternatives with
     * a categorical draw */  
    parallel for (n:Integer in 1..N) {
      f[n] <- f0[a[n]];
      if (f[n]?) {
        cpp {{
        ++P;
        }}
      } else {
        stderr.print("error: particles terminated prematurely.\n");
        exit(1);
      }
      while (f[n]!.w == -inf) {
        f[n] <- f0[ancestor(w0)];
        if (f[n]?) {
          cpp {{
          ++P;
          }}
        } else {
          stderr.print("error: particles terminated prematurely.\n");
          exit(1);
        }
      }
      w[n] <- f[n]!.w;
    }

    cpp {{
    P_ = P;
    }}

    /* propagate and weight until one further acceptance, that is discarded
     * for unbiasedness in the normalizing constant estimate */
    f1:Model! <- f0[ancestor(w0)];
    if (f1?) {
      P <- P + 1;
    } else {
      stderr.print("error: particles terminated prematurely.\n");
      exit(1);
    }
    while (f1!.w == -inf) {
      f1 <- f0[ancestor(w0)];
      if (f1?) {
        P <- P + 1;
      } else {
        stderr.print("error: particles terminated prematurely.\n");
        exit(1);
      }
    }
    
    /* update normalizing constant estimate */
    W:Real <- log_sum_exp(w);
    w <- w - (W - log(P - 1));
    if (t > 1) {
      Z[t] <- Z[t - 1] + (W - log(P - 1));
    } else {
      Z[t] <- W - log(P - 1);
    }
    
    /* diagnostics */
    propagations[t] <- P;
  }

  function diagnose(writer:Writer?) {
    super.diagnose(writer);
    if (writer?) {
      writer!.setIntegerVector("propagations", propagations);
    }
  }
}
