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
  P:List<Integer>;

  function initialize() {
    super.initialize();
    trigger <- 1.0;  // always resample
    P.clear();
  }
  
  function start() {
    super.start();
    P.pushBack(N);
  }

  function step() {          
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
    auto x0 <- x;
    auto w0 <- w;
    parallel for auto n in 1..N {
      auto a' <- a[n];
      auto x' <- clone<ForwardModel>(x0[a']);
      auto w' <- handle(x'.step());
      cpp {{
      ++P;
      }}
      while w' == -inf {
        /* keep trying until positive weight */
        a' <- ancestor(w0);
        x' <- clone<ForwardModel>(x0[a']);
        w' <- handle(x'.step());
        cpp {{
        ++P;
        }}
      }
      x[n] <- x';
      w[n] <- w[n] + w';
      a[n] <- a';
    }

    /* propagate and weight until one further acceptance, which is discarded
     * for unbiasedness in the normalizing constant estimate */
    auto a' <- ancestor(w0);
    auto x' <- clone<ForwardModel>(x0[a']);
    auto w' <- handle(x'.step());
    cpp {{
    ++P;
    }}
    while w' == -inf {
      /* keep trying until positive weight */
      a' <- ancestor(w0);
      x' <- clone<ForwardModel>(x0[a']);
      w' <- handle(x'.step());
      cpp {{
      ++P;
      }}
    }
    
    /* update propagations */
    Q:Integer;
    cpp{{
    Q_ = P;
    }}
    P.pushBack(Q);
  }
  
  function reduce() {
    super.reduce();
    
    /* correct normalizing constant estimate for rejections */
    auto Z <- this.Z.back();
    auto P <- this.P.back();
    this.Z.popBack();
    this.Z.pushBack(Z + log(N) - log(P - 1));
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("propagations", P);
  }
}
