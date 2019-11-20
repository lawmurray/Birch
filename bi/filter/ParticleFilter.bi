/**
 * Particle filter.
 */
class ParticleFilter {
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

  fiber filter(model:ForwardModel) -> (ForwardModel[_], Real[_], Real, Real) {
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
    
    auto t <- 0;
    while true {
      t <- t + 1;
    
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

  function read(buffer:Buffer) {
    nparticles <-? buffer.get("nparticles", nparticles);
    trigger <-? buffer.get("trigger", trigger);
    delayed <-? buffer.get("delayed", delayed);
  }

  function write(buffer:Buffer) {
    buffer.set("nparticles", nparticles);
    buffer.set("trigger", trigger);
    buffer.set("delayed", delayed);
  }
}
