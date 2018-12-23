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
   * Log-evidence.
   */
  Z:Real <- 0.0;
  
  /**
   * For each checkpoint, the effective sample size (ESS).
   */
  ess:List<Real>;
  
  /**
   * For each checkpoint, was resampling performed?
   */
  resample:List<Boolean>;
  
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
      auto t <- 0;
      if (verbose) {
        stderr.print("checkpoints:");
      }
      if ((!ncheckpoints? || t < ncheckpoints!) && start(m)) {
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
    
      /* choose single sample to yield */
      auto b <- ancestor(w);
      if (b > 0) {
        yield (s[b], Z);
      } else {
        error("particle filter degenerated.");
      }
    }
  }

  /**
   * Start the filter.
   *
   * Returns: Are particles yet to terminate?
   */  
  function start(m:Model) -> Boolean {
    if (length(s) != nparticles) {
      f1:(Model,Real)![nparticles];
      f <- f1;
      s1:Model[nparticles] <- m;
      s <- s1;
      w <- vector(0.0, nparticles);
    }
    Z <- 0.0;
    ess.clear();
    resample.clear();
    evidence.clear();

    /* this is a workaround at present for problems with nested clones: clone
     * into a local variable first, then copy into the member variable */
    f0:(Model,Real)! <- particle(m);
    f1:(Model,Real)![nparticles];
    for n:Integer in 1..nparticles {
      f1[n] <- clone<(Model,Real)!>(f0);
    }
    f <- f1;
    
    auto continue <- true;
    for (n:Integer in 1..nparticles) {
      if (f[n]?) {
        (s[n], w[n]) <- f[n]!;
      } else {
        continue <- false;
      }
    }
    if (continue) {
      auto W <- log_sum_exp(w);
      w <- w - (W - log(nparticles));
      Z <- Z + (W - log(nparticles));
    }
    
    return continue;
  }
  
  /**
   * Step to the next checkpoint.
   *
   * Returns: Are particles yet to terminate?
   */
  function step() -> Boolean {
    /* resample (if triggered) */
    ess.pushBack(global.ess(w));
    if (!(ess.back() > 0.0)) {  // may be nan
      error("particle filter degenerated.");
    }
    resample.pushBack(ess.back() < trigger*nparticles);
    if (resample.back()) {
      auto a <- ancestors(w);
      auto g <- f;
      for (n:Integer in 1..nparticles) {
        f[n] <- clone<(Model,Real)!>(g[a[n]]);
        w[n] <- 0.0;
      }
    }

    /* propagate and weight */
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
    
    if (continue) {
      /* update normalizing constant estimate */
      auto W <- log_sum_exp(w);
      w <- w - (W - log(nparticles));
      Z <- Z + (W - log(nparticles));
      evidence.pushBack(Z);
    }
    
    return continue;
  }
  
  /**
   * Finish the filter.
   */
  function finish() {
    //
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
    buffer.set("resample", resample);
    buffer.set("levidence", evidence);
  }
}

/*
 * Particle.
 */
fiber particle(m:Model) -> (Model, Real) {
  auto f <- m.simulate();
  if (f?) {
    yield (m, f!);
    while (f?) {
      yield (m, f!);
    }
  } else {
    /* ensure that even with no observations, the particle yields at least
     * once */
    yield (m, 0.0);
  }
}
