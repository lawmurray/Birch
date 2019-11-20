/**
 * Particle filter.
 */
class ParticleFilter {
  /**
   * Model.
   */
  model:ForwardModel;

  /**
   * Number of steps.
   */
  nsteps:Integer <- 1;

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

  fiber filter() -> (ForwardModel[_], Real[_], Real, Real) {
    auto x <- clone<ForwardModel>(model, nparticles);  // particles
    auto w <- vector(0.0, 0);  // log-weights
    auto ess <- 0.0;  // effective sample size
    auto levidence <- 0.0;  // incremental log-evidence
    
    /* event handler */
    h:Handler <- play;
    if delayed {
      h <- global.delay;
    }

    /* initialize and weight */
    parallel for n in 1..nparticles {
      w[n] <- h.handle(x[n].simulate());
    }
    (ess, levidence) <- resample_reduce(w);
    yield (x, w, ess, levidence);
      
    for t in 1..nsteps {
      /* resample */
      if ess <= trigger*nparticles {
        auto a <- resample_systematic(w);
        dynamic parallel for n in 1..nparticles {
          if a[n] != n {
            x[n] <- clone<ForwardModel>(x[a[n]]);
          }
        }
      }
      
      /* propagate and weight */
      parallel for n in 1..nparticles {
        w[n] <- w[n] + h.handle(x[n].simulate(t));
      }
      (ess, levidence) <- resample_reduce(w);
      yield (x, w, ess, levidence);
    }
  }

  function setModel(model:ForwardModel) {
    this.model <- model;
    nsteps <- model.size();
  }

  function read(buffer:Buffer) {
    nsteps <-? buffer.get("nsteps", nsteps);
    nparticles <-? buffer.get("nparticles", nparticles);
    trigger <-? buffer.get("trigger", trigger);
    delayed <-? buffer.get("delayed", delayed);
  }

  function write(buffer:Buffer) {
    buffer.set("nsteps", nsteps);
    buffer.set("nparticles", nparticles);
    buffer.set("trigger", trigger);
    buffer.set("delayed", delayed);
  }
}
