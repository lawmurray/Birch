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
  f:(Model, Real)![_];
  
  /**
   * Samples.
   */
  s:Model[_];
  
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
      auto t <- 0;
      if (verbose) {
        stderr.print("checkpoints:");
      }
      if ((!ncheckpoints? || t < ncheckpoints!) && start()) {
        t <- t + 1;
        if (verbose) {
          stderr.print(" " + t);
        }
      }
      while ((!ncheckpoints? || t < ncheckpoints!) && step()) {
        t <- t + 1;
        if (verbose) {
          stderr.print(" " + t);
        }
      }
      if (ncheckpoints? && t != ncheckpoints!) {
        error("particles terminated after " + t + " checkpoints, at least " + ncheckpoints! + " expected.");
      }
      if (verbose) {
        stderr.print(", log evidence: " + Z + "\n");
      }
      finish();
      
      yield (s[b], Z);
    }
  }

  /**
   * Initialize.
   */  
  function initialize(m:Model) {
    s1:Model[nparticles] <- m;
    s <- s1;
    w <- vector(0.0, nparticles);
    a <- vector(0, nparticles);
    Z <- 0.0;
    ess.clear();
    r.clear();
    evidence.clear();

    /* this is a workaround at present for problems with nested clones: clone
     * into a local variable first, then copy into the member variable */
    f1:(Model,Real)![nparticles];
    auto f0 <- particle(m);
    for n:Integer in 1..nparticles {
      f1[n] <- clone<(Model,Real)!>(f0);
    }
    f <- f1;
  }
  
  /**
   * Advance to the first checkpoint.
   *
   * Returns: Are particles yet to terminate?
   */
  function start() -> Boolean {
    auto continue <- propagate();
    if (continue) {
      reduce();
    }
    return continue;
  }
  
  /**
   * Advance to the next checkpoint.
   *
   * Returns: Are particles yet to terminate?
   */
  function step() -> Boolean {
    r.pushBack(ess.back() < trigger*nparticles);
    if (r.back()) {
      resample();
      copy();
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
    auto f1 <- f;
    auto f2 <- f;
    for n:Integer in 1..nparticles {
      f2[n] <- clone<(Model,Real)!>(f1[a[n]]);
    }
    f <- f2;
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
      if (f[n]?) {
        v:Real;
        (s[n], v) <- f[n]!;
        w[n] <- w[n] + v;
      } else {
        continue <- false;
      }
    }
    return continue;
  }
  
  /**
   * Compute any required summary statistics.
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

/*
 * Particle.
 */
fiber particle(m:Model) -> (Model, Real) {
  auto f <- m.simulate();
  while (f?) {
    yield (m, f!);
  }
}
