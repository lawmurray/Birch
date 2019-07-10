/**
 * Particle filter.
 */
class ParticleFilter < ForwardSampler {  
  /**
   * Particles.
   */
  x:ForwardModel[_];
  
  /**
   * Log-weights.
   */
  w:Real[_];

  /**
   * Ancestor indices.
   */
  a:Integer[_];
  
  /**
   * Index of the chosen sample.
   */
  b:Integer <- 1;
  
  /**
   * Number of particles.
   */
  N:Integer <- 1;
    
  /**
   * Threshold for resampling. Resampling is performed whenever the
   * effective sample size, as a proportion of `N`, drops below this
   * threshold.
   */
  trigger:Real <- 0.7;
  
  /**
   * For each checkpoint, the logarithm of the normalizing constant estimate
   * so far.
   */
  Z:Queue<Real>;
  
  /**
   * For each checkpoint, the effective sample size (ESS).
   */
  ess:Queue<Real>;
  
  /**
   * At each checkpoint, how much memory is in use?
   */
  memory:Queue<Integer>;
  
  /**
   * At each checkpoint, what is the elapsed wallclock time?
   */
  elapsed:Queue<Real>; 

  fiber sample() -> (Model, Real) {
    for auto n in 1..nsamples {
      if verbose && T > 0 {
        stderr.print("steps:");
      }
      initialize();
      start();
      reduce();
      for auto t in 1..T {
        if verbose {
          stderr.print(" " + t);
        }
        resample();
        step();
        reduce();
      }
      finish();
      if verbose {
        if T > 0 {
          stderr.print(", ");
        }
        stderr.print("log weight: " + sum(Z.walk()) + "\n");
      }
      finalize();
      yield (clone<ForwardModel>(x[b]), sum(Z.walk()));
    }
  }

  /**
   * Initialize.
   */
  function initialize() {
    Z.clear();
    ess.clear();
    memory.clear();
    elapsed.clear();

    w <- vector(0.0, N);
    a <- iota(1, N);
    x1:Vector<ForwardModel>;
    x1.enlarge(N, clone<ForwardModel>(archetype!));
    x <- x1.toArray();
    dynamic parallel for auto n in 1..N {
      x[n] <- clone<ForwardModel>(x[n]);
    }
    tic();
  }
  
  /**
   * Start particles.
   */
  function start() {
    parallel for auto n in 1..N {
      w[n] <- w[n] + x[n].start();
    }
  }
  
  /**
   * Step particles.
   */
  function step() {
    parallel for auto n in 1..N {
      w[n] <- w[n] + x[n].step();
    }
  }

  /**
   * Compute summary statistics.
   */
  function reduce() {
    auto m <- max(w);
    auto W <- 0.0;
    auto W2 <- 0.0;
    
    for auto n in 1..N {
      auto v <- exp(w[n] - m);
      W <- W + v;
      W2 <- W2 + v*v;
    }
    auto V <- log(W) + m - log(N);
    w <- w - V;  // normalize weights to sum to N
   
    /* effective sample size */
    ess.pushBack(W*W/W2);
  
    /* normalizing constant estimate */
    Z.pushBack(V);
    memory.pushBack(memoryUse());
    elapsed.pushBack(toc());
  }

  /**
   * Resample particles.
   */
  function resample() {
    if isTriggered() {
      /* resample */
      a <- global.resample(w);
      w <- vector(0.0, N);
      
      /* copy particles */
      dynamic parallel for auto n in 1..N {
        if a[n] == n {
          // avoid clone overhead
        } else {
          x[n] <- clone<ForwardModel>(x[a[n]]);
        }
      }
    } else {
      a <- iota(1, N);
    }
  }

  /**
   * Finish.
   */
  function finish() {
    b <- max(1, ancestor(w));
  }
  
  /**
   * Finalize.
   */
  function finalize() {
    //
  }
  
  /**
   * Given the current state, should resampling be performed?
   */
  function isTriggered() -> Boolean {
    return ess.back() <= trigger*N;
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    N <-? buffer.get("nparticles", N);
    trigger <-? buffer.get("trigger", trigger);
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("nparticles", N);
    buffer.set("trigger", trigger);
    buffer.set("levidence", Z);
    buffer.set("ess", ess);
    buffer.set("elapsed", elapsed);
    buffer.set("memory", memory);
  }
}
