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
  P:Vector<Integer>;

  function start(m:Model) {
    super.start(m);
    trigger <- 1.0;  // always resample
    P.clear();
  }

  function step() -> Boolean {
    r.pushBack(true);
    auto continue <- propagate();
    if continue {
      reduce();
    }
    return continue;
  }

  function propagate() -> Boolean {
    /* as `parallel for` is used below, an atomic is necessary to accumulate
     * the total number of propagations; nested C++ is required for this at
     * this stage */
    cpp {{
    std::atomic<int> P;
    P = 0;
    }}

    /* propagate and weight until `nparticles` have been accepted; the first
     * `nparticles` proposals are drawn using the standard resampler; as each
     * is rejected it is replaced with a categorical draw until acceptance */  
    auto f0 <- f;
    auto W0 <- cumulative_weights(w);
    a <- cumulative_offspring_to_ancestors(systematic_cumulative_offspring(W0));
    w <- vector(0.0, nparticles);
    auto continue <- true;
    parallel for n:Integer in 1..nparticles {
      do {
        f[n] <- clone<(Model,Real)!>(f0[a[n]]);
        if f[n]? {
          (x[n], w[n]) <- f[n]!;
          cpp {{
          ++P;
          }}
          if w[n] == -inf {
            /* replace with a categorical draw for next attempt */
            a[n] <- cumulative_ancestor(W0);
          }
        } else {
          continue <- false;
        }
      } while continue && w[n] == -inf;
    }

    if (continue) {
      /* propagate and weight until one further acceptance, that is discarded
       * for unbiasedness in the normalizing constant estimate */
     x1:Model?;
     w1:Real;
     do {
        auto f1 <- clone<(Model,Real)!>(f0[cumulative_ancestor(W0)]);
        if f1? {
          (x1, w1) <- f1!;
          cpp {{
          ++P;
          }}
        } else {
          continue <- false;
        }
      } while (continue && w1 == -inf);
    }

    /* update propagations */
    P:Integer;
    cpp {{
    P_ = P;
    }}
    this.P.pushBack(P);

    return continue;
  }

  function reduce() {
    /* effective sample size */
    e.pushBack(ess(w));
    if (!(e.back() > 0.0)) {  // > 0.0 as may be nan
      error("particle filter degenerated.");
    }

    /* normalizing constant estimate */
    auto W <- log_sum_exp(w);
    auto P <- this.P.back();
    auto Z <- W - log(P - 1);
    w <- w - Z;
    if (this.Z.empty()) {
      this.Z.pushBack(Z);
    } else {
      this.Z.pushBack(this.Z.back() + Z);
    }
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("propagations", P);
  }
}
