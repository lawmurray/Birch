/**
 * Particle filter.
 *
 * The ParticleFilter class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Filter.svg"></object>
 * </center>
 */
class ParticleFilter {
  /**
   * Particles.
   */
  x:Particle[_];

  /**
   * Log weights.
   */
  w:Real[_];

  /**
   * Ancestor indices.
   */
  a:Integer[_];

  /**
   * Effective sample size.
   */
  ess:Real <- 0.0;
  
  /**
   * Logarithm of sum of weights.
   */
  lsum:Real <- 0.0;

  /**
   * Log normalizing constant.
   */
  lnormalize:Real <- 0.0;
  
  /**
   * Number of propagations. This is not the same as the number of particles;
   * the number of propagations performed may, according to the filter type,
   * differ from the number of particles, such as for alive and rejection
   * control particle filters.
   */
  npropagations:Integer <- 0;
  
  /**
   * Accept rate of moves.
   */
  raccept:Real <- 0.0;

  /**
   * Number of steps. If this has no value, the model will be required to
   * suggest an appropriate value.
   */
  nsteps:Integer?;

  /**
   * Number of additional forecast steps per step.
   */
  nforecasts:Integer <- 0;
  
  /**
   * Number of particles.
   */
  nparticles:Integer <- 1;

  /**
   * Threshold for resampling. Resampling is performed whenever the
   * effective sample size, as a proportion of `N`, drops below this
   * threshold.
   */
  trigger:Real <- 0.7;
  
  /**
   * Should delayed sampling be used?
   */
  delayed:Boolean <- true;

  /**
   * Size. This is the number of steps of `filter(Model, Integer)` to be
   * performed after the initial call to `filter(Model)`. Note that
   * `filter(Model, Integer)` must be called before `size()`.
   */
  function size() -> Integer {
    assert nsteps?;
    return nsteps!;
  }
  
  /**
   * Create a particle of the type required for this filter.
   */
  function particle(archetype:Model) -> Particle {
    return Particle(archetype);
  }
  
  /**
   * Initialize filter.
   *
   * - archetype: Archetype. This is an instance of the appropriate model
   *   class that may have one more random variables fixed to known values,
   *   representing the inference problem (or target distribution).
   */
  function filter(archetype:Model) {
    x <- clone(particle(archetype), nparticles);
    w <- vector(0.0, nparticles);
    a <- iota(1, nparticles);
    ess <- nparticles;
    lsum <- 0.0;
    lnormalize <- 0.0;
    npropagations <- nparticles;
    
    /* size */
    if !nsteps? {
      nsteps <- archetype.size();
    }
    
    start();
    reduce();
  }

  /**
   * Filter one step.
   *
   * - t: The step number, beginning at 0.
   */
  function filter(t:Integer) {
    resample(t);
    step(t);
    reduce();
  }
  
  /**
   * Start particles.
   */
  function start() {
    auto play <- PlayHandler(delayed);
    parallel for n in 1..nparticles {
      w[n] <- play.handle(x[n].m.simulate());
    }  
  }
  
  /**
   * Step particles.
   */
  function step(t:Integer) {
    auto play <- PlayHandler(delayed);
    parallel for n in 1..nparticles {
      w[n] <- w[n] + play.handle(x[n].m.simulate(t));
    }
  }

  /**
   * Forecast particles.
   */
  function forecast(t:Integer) {
    auto play <- PlayHandler(delayed);
    parallel for n in 1..nparticles {
      w[n] <- w[n] + play.handle(x[n].m.forecast(t));
    }
  }
  
  /**
   * Compute reductions, such as effective sample size and normalizing
   * constant estimate.
   */
  function reduce() {
    (ess, lsum) <- resample_reduce(w);
    lnormalize <- lnormalize + lsum - log(Real(nparticles));
  }
  
  /**
   * Resample particles.
   */
  function resample(t:Integer) {
    if ess <= trigger*nparticles {
      collect();
      a <- resample_systematic(w);
      w <- vector(0.0, nparticles);
      dynamic parallel for n in 1..nparticles {
        if a[n] != n {
          x[n] <- clone(x[a[n]]);
        }
      }
      collect();
    } else {
      /* normalize weights to sum to nparticles */
      w <- w - vector(lsum - log(Real(nparticles)), nparticles);
    }
  }

  /**
   * Write only the current state to a buffer.
   */
  function write(buffer:Buffer, t:Integer) {
    buffer.set("sample", clone(x));
    buffer.set("lweight", w);
    buffer.set("lnormalize", lnormalize);
    buffer.set("ess", ess);
    buffer.set("npropagations", npropagations);
    buffer.set("raccept", raccept);
  }

  override function read(buffer:Buffer) {
    super.read(buffer);
    nsteps <-? buffer.get("nsteps", nsteps);
    nforecasts <-? buffer.get("nforecasts", nforecasts);
    nparticles <-? buffer.get("nparticles", nparticles);
    trigger <-? buffer.get("trigger", trigger);
    delayed <-? buffer.get("delayed", delayed);
  }

  override function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("nsteps", nsteps!);
    buffer.set("nforecasts", nforecasts);
    buffer.set("nparticles", nparticles);
    buffer.set("trigger", trigger);
    buffer.set("delayed", delayed);
  }
}
