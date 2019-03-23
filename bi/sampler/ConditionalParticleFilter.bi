/**
 * Conditional particle filter. Conditions on a previously known path
 * $x'_{1:T}$.
 */
class ConditionalParticleFilter < ParticleFilter {
  /**
   * Conditioned path, drawn from previous iteration.
   */
  x0:List<Model>;
  
  /**
   * Conditioned path weights, drawn from previous iteration.
   */
  w0:List<Real>;
  
  /**
   * All paths, in current iteration.
   */
  X:List<Model>[_];

  /**
   * All weights, in current iteration.
   */
  W:List<Real>[_];
   
  function start(m:Model) {
    super.start(m);
    X1:List<Model>[nparticles];
    W1:List<Real>[nparticles];
    X <- X1;
    W <- W1;
  }

  function resample() {
    super.resample();
    ///@todo Do this properly with conditional distribution for resampler
    if !x0.empty() {
      a[nparticles] <- nparticles;
    }
  }
  
  function copy() {
    super.copy();
    auto X0 <- X;
    auto W0 <- W;
    for n:Integer in 1..nparticles {
      X[n] <- clone<List<Model>>(X0[a[n]]);
      W[n] <- clone<List<Real>>(W0[a[n]]);
    }
  }

  function propagate() -> Boolean {
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

  function finish() {
    super.finish();
    x0 <- X[b];
    w0 <- W[b];
  }
}
