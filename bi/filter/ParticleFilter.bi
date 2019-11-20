/**
 * Particle filter.
 */
class ParticleFilter {
  /**
   * The archetype.
   */
  archetype:ForwardModel?;

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

  fiber filter() -> (ForwardModel[_], Real[_], Real, Real) {
    assert archetype?;
    
    auto x <- clone<ForwardModel>(archetype!, nparticles);  // particles
    auto w <- vector(0.0, nparticles);  // log-weights
    auto ess <- 1.0*nparticles;  // effective sample size
    auto levidence <- 0.0;  // incremental log-evidence

    for t in 0..nsteps {
      /* resample */
      if ess <= trigger*nparticles {
        auto a <- resample_systematic(w);
        dynamic parallel for n in 1..nparticles {
          if a[n] != n {
            x[n] <- clone<ForwardModel>(x[a[n]]);
          }
        }
      }
      
      if t == 0 {
        /* initialize and weight */
        parallel for n in 1..nparticles {
          w[n] <- delay.handle(x[n].simulate());
        }
      } else {
        /* propagate and weight */
        parallel for n in 1..nparticles {
          w[n] <- w[n] + delay.handle(x[n].simulate(t));
        }
      }
      
      /* ESS and incremental evidence */
      (ess, levidence) <- resample_reduce(w);

      yield (x, w, ess, levidence);
    }
  }

  function setArchetype(archetype:Model) {
    this.archetype <-? ForwardModel?(archetype);
    if !this.archetype? {
      error("model class must be a subtype of ForwardModel to use ParticleFilter.");
    }
    nsteps <- this.archetype!.size();
  }

  function read(buffer:Buffer) {
    nsteps <-? buffer.get("nsteps", nsteps);
    nparticles <-? buffer.get("nparticles", nparticles);
    trigger <-? buffer.get("trigger", trigger);
  }

  function write(buffer:Buffer) {
    buffer.set("nsteps", nsteps);
    buffer.set("nparticles", nparticles);
    buffer.set("trigger", trigger);
  }
}
