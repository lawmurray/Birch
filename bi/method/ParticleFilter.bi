/**
 * Particle filter. Performs a bootstrap particle filter in the most basic
 * case, or where conjugacy relationships are used by the model, an auxiliary
 * or Rao--Blackwellized particle filter.
 */
class ParticleFilter < Sampler {
  /**
   * Number of particles.
   */
  nparticles:Integer <- 1;
  
  /**
   * Threshold for resampling. Resampling is performed whenever the
   * effective sample size, as a proportion of `nparticles`, drops below this
   * threshold.
   */
  trigger:Real <- 0.7;

  /**
   * Particles.
   */
  x:Model[_];
  
  /**
   * Log-weights.
   */
  w:Real[_];

  /**
   * Ancestor indices.
   */
  a:Integer[_];
  
  /**
   * Log-evidence.
   */
  Z:Real <- 0.0;
  
  /**
   * Index of chosen path at end of filter.
   */
  b:Integer <- 0;
  
  /**
   * For each checkpoint, the effective sample size (ESS).
   */
  ess:List<Real>;
  
  /**
   * For each checkpoint, was resampling performed?
   */
  r:List<Boolean>;
  
  /**
   * For each checkpoint, the logarithm of the normalizing constant estimate
   * up to that checkpoint.
   */
  evidence:List<Real>;
  
  fiber sample(m:Model) -> (Model, Real) {
    /* if a number of checkpoints hasn't been explicitly provided, compute
     * this from the model (may still be unknown by the model, too) */
    if (!ncheckpoints?) {
      ncheckpoints <- m.checkpoints();
    }

    /* sample */  
    for (i:Integer in 1..nsamples) {
      initialize(m);
      start();
      if (verbose) {
        stderr.print("checkpoints:");
      }
      auto t <- 0;
      while ((!ncheckpoints? || t < ncheckpoints!) && step()) {
        t <- t + 1;
        if (verbose) {
          stderr.print(" " + t);
        }
      }
      if (ncheckpoints? && t != ncheckpoints!) {
        error("particles terminated after " + t + " checkpoints, but " + ncheckpoints! + " requested.");
      }
      if (verbose) {
        stderr.print(", log evidence: " + Z + "\n");
      }
      finish();
      yield (x[b], Z);
    }
  }

  /**
   * Initialize.
   */  
  function initialize(m:Model) {
    w <- vector(0.0, nparticles);
    Z <- 0.0;
    ess.clear();
    r.clear();
    evidence.clear();

    /* this is a workaround at present for problems with nested clones: clone
     * into a local variable first, then copy into the member variable */
    x1:Model[nparticles];
    a1:Integer[nparticles];
    for n:Integer in 1..nparticles {
      x1[n] <- clone<Model>(m);
      a1[n] <- n;
    }
    x <- x1;
    a <- a1;
  }
  
  /**
   * Start particles.
   */
  function start() {
    parallel for (n:Integer in 1..nparticles) {
      x[n].start();
    }
  }
  
  /**
   * Step particles to the next checkpoint.
   */
  function step() -> Boolean {
    if (!ess.empty()) {
      r.pushBack(ess.back() < trigger*nparticles);
      if (r.back()) {
        resample();
        copy();
      }
    }
    auto continue <- propagate();
    if (continue) {
      reduce();
    }
    return continue;
  }
  
  /**
   * Resample particles.
   */
  function resample() {
    a <- ancestors(w);
  }

  /**
   * Copy particles after resampling.
   */
  function copy() {
    /* this is a workaround at present for problems with nested clones: clone
     * into local variables first, then update member variables */
    auto x1 <- x;
    for n:Integer in 1..nparticles {
      x1[n] <- clone<Model>(x[a[n]]);
    }
    x <- x1;
    for n:Integer in 1..nparticles {
      w[n] <- 0.0;
    }
  }
  
  /**
   * Propagate particles.
   */
  function propagate() -> Boolean {
    auto continue <- true;
    parallel for (n:Integer in 1..nparticles) {
      auto v <- x[n].step();
      if v? {
        w[n] <- w[n] + v!;
      } else {
        continue <- false;      
      }
    }
    return continue;
  }
  
  /**
   * Compute summary statistics.
   */
  function reduce() {
    /* effective sample size */
    ess.pushBack(global.ess(w));
    if (!(ess.back() > 0.0)) {  // may be nan
      error("particle filter degenerated.");
    }
  
    /* normalizing constant estimate */
    auto W <- log_sum_exp(w);
    w <- w - (W - log(nparticles));
    Z <- Z + (W - log(nparticles));
    evidence.pushBack(Z);
  }  

  /**
   * Finish the filter.
   */
  function finish() {
    /* choose single sample to yield */
    b <- ancestor(w);
    if (b <= 0) {
      error("particle filter degenerated.");
    }
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    auto nparticles1 <- buffer.getInteger("nparticles");
    if nparticles1? {
      nparticles <- nparticles1!;
    }
    auto trigger1 <- buffer.getReal("trigger");
    if trigger1? {
      trigger <- trigger1!;
    }
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("nparticles", nparticles);
    buffer.set("trigger", trigger);
    buffer.set("ess", ess);
    buffer.set("resample", r);
    buffer.set("levidence", evidence);
  }
}
