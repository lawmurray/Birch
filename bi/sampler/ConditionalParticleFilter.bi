/**
 * Conditional particle filter.
 */
class ConditionalParticleFilter<Handler> < ParticleFilter<Handler> {   
  function resample() {
    super.resample();
    ///@todo Do this properly with conditional distribution for resampler
    if !x0.empty() {
      a[nparticles] <- nparticles;
    }
  }

  function step() -> Boolean {
    auto continue <- true;
    auto N <- nparticles;
    
    if !x0.empty() {
      /* condition on the given path */
      x[N] <- x0.front();
      w[N] <- w[N] + w0.front();
      X[N].pushBack(x0.front());
      W[N].pushBack(w0.front());
      x0.popFront();
      w0.popFront();
      N <- N - 1;
    }
    parallel for (n:Integer in 1..N) {
      w1:Real;
      if f[n]? {
        (x[n], w1) <- f[n]!;
        w[n] <- w[n] + w1;
        X[n].pushBack(x[n]);
        W[n].pushBack(w1);
      } else {
        continue <- false;      
      }
    }
    return continue;
  }
}
