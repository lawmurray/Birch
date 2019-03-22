/**
 * Particle filter. Performs a bootstrap particle filter in the most basic
 * case, or where conjugacy relationships are used by the model, an auxiliary
 * or Rao--Blackwellized particle filter.
 */
class ParticleFilter < Sampler {
  /**
   * Particles.
   */
  x:Particle[_];
  
  /**
   * Log-weights.
   */
  w:Real[_];

  /**
   * Ancestor indices.
   */
  a:Integer[_];
  
  /**
   * Index of the chosen path at the end of the filter.
   */
  b:Integer <- 0;
  
  /**
   * For each checkpoint, the logarithm of the normalizing constant estimate
   * so far.
   */
  Z:Vector<Real>;
  
  /**
   * For each checkpoint, the effective sample size (ESS).
   */
  e:Vector<Real>;
  
  /**
   * For each checkpoint, was resampling performed?
   */
  r:Vector<Boolean>;
  
  /**
   * At each checkpoint, how much memory is in use?
   */
  memory:Vector<Integer>;
  
  /**
   * At each checkpoint, what is the elapsed wallclock time?
   */
  elapsed:Vector<Real>; 
    
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

  function sample(m:Model) -> (Model, Real) {
    /* if a number of checkpoints hasn't been explicitly provided, compute
     * this from the model (may still be unknown by the model, too) */
    if !ncheckpoints? {
      ncheckpoints <- m.checkpoints();
    }

    start(m);
    if verbose {
      stderr.print("checkpoints:");
    }
    auto t <- 0;
    while (!ncheckpoints? || t < ncheckpoints!) && step() {
      t <- t + 1;
      if verbose {
        stderr.print(" " + t);
      }
    }
    if ncheckpoints? && t != ncheckpoints! {
      error("particles terminated after " + t + " checkpoints, but " + ncheckpoints! + " requested.");
    }
    if verbose && !Z.empty() {
      stderr.print(", log evidence: " + Z.back() + "\n");
    }
    finish();
      
    w:Real <- 0.0;
    if !Z.empty() {
      w <- Z.back();
    }
    return (x[b], w);
  }
  
  /**
   * Start.
   */  
  function start(m:Model) {
    Z.clear();
    e.clear();
    r.clear();
    memory.clear();
    elapsed.clear();
    tic();
    x0:Particle(m);
    x1:Vector<Particle>;
    parallel for auto n in 1..nparticles {
      x1.pushBack(clone<Particle>(x0));
    }
    x <- x1;
    w <- vector(0.0, nparticles);
    a <- iota(1, nparticles);
  }
  
  /**
   * Step particles to the next checkpoint.
   */
  function step() -> Boolean {
    if !e.empty() {
      r.pushBack(e.back() < trigger*nparticles);
      if r.back() {
        resample();
        copy();
      }
    }
    auto continue <- propagate();
    if continue {
      reduce();
    }
    return continue;
  }
  
  /**
   * Resample particles.
   */
  function resample() {
    a <- permute_ancestors(ancestors(w));
    w <- vector(0.0, nparticles);
  }

  /**
   * Copy particles after resampling.
   */
  function copy() {
    /* there are couple of different strategies here: permute_ancestors()
     * would avoid the use of the temporary f0, but f0 ensures that particles
     * with the same ancestor are contiguous in f after the copy, which is
     * more cache efficient */
    auto x0 <- x;
    parallel for n:Integer in 1..nparticles {
      x[n] <- clone<Particle>(x0[a[n]]);
    }
  }
    
  /**
   * Propagate particles.
   */
  function propagate() -> Boolean {
    auto continue <- true;
    parallel for auto n in 1..nparticles {
      if continue {
        w[n] <- w[n] + x[n].step();
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
    W:Real <- 0.0;
    W2:Real <- 0.0;
    m:Real <- max(w);
    v:Real;
    
    for auto n in 1..length(w) {
      v <- exp(w[n] - m);
      W <- W + v;
      W2 <- W2 + v*v;
    }
    auto ess <- W*W/W2;
    auto Z <- log(W) + m - log(nparticles);
    w <- w - Z;  // normalize weights to sum to nparticles
   
    /* effective sample size */
    e.pushBack(ess);
    if !(e.back() > 0.0) {  // > 0.0 as may be nan
      error("particle filter degenerated.");
    }
  
    /* normalizing constant estimate */
    if this.Z.empty() {
      this.Z.pushBack(Z);
    } else {
      this.Z.pushBack(this.Z.back() + Z);
    }
    elapsed.pushBack(toc());
    memory.pushBack(memoryUse());
  }

  /**
   * Finish the filter.
   */
  function finish() {
    b <- ancestor(w);
    if b <= 0 {
      error("particle filter degenerated.");
    }
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    nparticles <-? buffer.get("nparticles", nparticles);
    trigger <-? buffer.get("trigger", trigger);
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("nparticles", nparticles);
    buffer.set("trigger", trigger);
    buffer.set("levidence", Z);
    buffer.set("ess", e);
    buffer.set("resample", r);
    buffer.set("elapsed", elapsed);
    buffer.set("memory", memory);
  }
}
