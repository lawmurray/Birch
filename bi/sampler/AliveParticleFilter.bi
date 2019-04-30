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
  P:Queue<Integer>;

  function initialize() {
    super.initialize();
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
    auto x0 <- x;
    auto w0 <- w;
    parallel for auto n in 1..N {
      w[n] <- x[n].step();
      cpp {{
      ++P;
      }}
      while w[n] == -inf {  // repeat until weight is positive
        a[n] <- ancestor(w0);
        x[n] <- clone<ForwardModel>(x0[a[n]]);
        w[n] <- x[n].step();
        cpp {{
        ++P;
        }}
      }
    }

    /* propagate and weight until one further acceptance, which is discarded
     * for unbiasedness in the normalizing constant estimate */
    w':Real;
    do {
      auto a' <- ancestor(w0);
      auto x' <- clone<ForwardModel>(x0[a']);
      w' <- x'.step();
      cpp {{
      ++P;
      }}
    } while w' == -inf;  // repeat until weight is positive
    
    /* update propagations */
    Q:Integer;
    cpp{{
    Q = P;
    }}
    this.P.pushBack(Q);
  }
  
  function reduce() {
    super.reduce();
    
    /* correct normalizing constant estimate for rejections */
    auto Z <- this.Z.back();
    auto P <- this.P.back();
    this.Z.popBack();
    this.Z.pushBack(Z + log(N) - log(P - 1));
  }

  function resample() {
    /* just compute ancestors and offspring, don't copy */
    if isTriggered() {
      (a, o) <- global.resample(w);
      w <- vector(0.0, N);
    } else {
      a <- iota(1, N);
      o <- vector(1, N);
    }
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("propagations", P);
  }
}
