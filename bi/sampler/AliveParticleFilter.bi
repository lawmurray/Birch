/**
 * Alive particle filter. When propagating and weighting particles, the
 * alive particle filter maintains $N$ particles with non-zero weight, rather
 * than $N$ particles in total as with the standard particle filter.
 */
class AliveParticleFilter < ParticleFilter {
  /**
   * For each checkpoint, the number of propagations that were performed to
   * achieve $N+1$ acceptances.
   */
  P:Queue<Integer>;

  function initialize() {
    super.initialize();
    P.clear();
  }
  
  function start() {
    super.start();
    P.pushBack(N+1);
  }

  function step() {          
    P:Integer[N+1];
    auto x0 <- x;
    auto w0 <- w;
    parallel for n in 1..N+1 {
      if n <= N {
        x[n] <- clone<ForwardModel>(x0[a[n]]);
        P[n] <- 1;
        w[n] <- x[n].step();
        while w[n] == -inf {  // repeat until weight is positive
          a[n] <- ancestor(w0);
          x[n] <- clone<ForwardModel>(x0[a[n]]);
          P[n] <- P[n] + 1;
          w[n] <- x[n].step();
        }
      } else {
        /* propagate and weight until one further acceptance, which is discarded
         * for unbiasedness in the normalizing constant estimate */
        w':Real;
        P[n] <- 0;
        do {
          auto a' <- ancestor(w0);
          auto x' <- clone<ForwardModel>(x0[a']);
          P[n] <- P[n] + 1;
          w' <- x'.step();
        } while w' == -inf;  // repeat until weight is positive
      }
    }
    
    /* update propagations */
    this.P.pushBack(sum(P));
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
    /* just compute ancestors, don't copy or reset weights */
    a <- global.resample(w);
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("propagations", P);
  }
}
