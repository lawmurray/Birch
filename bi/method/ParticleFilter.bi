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
  f:Model![_];
  
  /**
   * Log-weights.
   */
  w:Real[_];
  
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
  
  fiber sample(m:Model) -> Model {
    for (s:Integer in 1..nsamples) {
      start(m);
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
        error("particles terminated after " + t + " checkpoints, at least " + ncheckpoints! + " expected.");
      }
      if (verbose) {
        stderr.print(", log evidence: " + evidence.back() + "\n");
      }
      finish();
    
      /* choose single sample to yield */
      auto b <- ancestor(w);
      if (b > 0) {
        f[b]!.w <- evidence.back();
        yield f[b]!;
      } else {
        error("particle filter degenerated.");
      }
    }
  }

  /**
   * Start the filter.
   */  
  function start(m:Model) {
    f0:Model! <- particle(m);
    f1:Model![nparticles];
    for n:Integer in 1..nparticles {
      f1[n] <- clone<Model!>(f0);
    }
    
    this.f <- f1;
    this.w <- vector(0.0, nparticles);
    
    ess.clear();
    resample.clear();
    evidence.clear();
  }
  
  /**
   * Step to the next checkpoint.
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
        f[n] <- clone<Model!>(g[a[n]]);
        w[n] <- 0.0;
      }
    }

    /* propagate and weight */
    auto continue <- true;
    parallel for (n:Integer in 1..nparticles) {
      if (f[n]?) {
        w[n] <- w[n] + f[n]!.w;
      } else {
        continue <- false;
      }
    }
    
    if (continue) {
      /* update normalizing constant estimate */
      auto W <- log_sum_exp(w);
      w <- w - (W - log(nparticles));
      if (evidence.empty()) {
        evidence.pushBack(W - log(nparticles));
      } else {
        evidence.pushBack(evidence.back() + W - log(nparticles));
      }
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
    buffer.set("evidence", evidence);
  }
}

/*
 * Particle.
 */
fiber particle(m:Model) -> Model {
  auto f <- m.simulate();
  while (f?) {
    m.w <- f!;
    yield m;
  }
}
