/**
 * Conditional particle filter.
 *
 * The ParticleFilter class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Filter.svg"></object>
 * </center>
 */
class ConditionalParticleFilter < ParticleFilter {
  /**
   * Trace of the reference particle. This will have no value for the first
   * iteration of the filter. Subsequent iterations will draw a particle from
   * the previous iteration to condition the new iteration, setting this
   * variable.
   */
  r:Trace?;

  /**
   * Chosen particle index. This is the index, at the final step, of the
   * particle chosen as a weighted sample from the target distribution,
   * used as the reference particle for the next iteration of the conditional
   * particle filter.
   */
  b:Integer <- 0;

  /**
   * Should ancestor sampling be used?
   */
  ancestor:Boolean <- false;

  /**
   * Will the enclosing sampler initialize the initial state of the reference
   * particle? This is set by ParticleGibbs, which initializes `x` externally
   * after the first iteration.
   */
  alreadyInitialized:Boolean <- false;

  /**
   * Replay event handler.
   */
  replay:TraceHandler <- global.replay;

  override function filter(archetype:Model) {
    x <- clone(archetype, nparticles);
    w <- vector(0.0, nparticles);
    a <- iota(1, nparticles);
    b <- 1;
    ess <- nparticles;
    lsum <- 0.0;
    lnormalize <- 0.0;
    npropagations <- nparticles;

    /* size */
    if !nsteps? {
      nsteps <- archetype.size();
    }
        
    /* event handlers */
    if delayed {
      play <- global.playDelay;
      replay <- global.replayDelay;
    } else {
      play <- global.play;
      replay <- global.replay;
    }

    /* initialize particles */
    if !alreadyInitialized {
      parallel for n in 1..nparticles {
        if r? && n == b {
          w[n] <- replay.handle(r!, x[n].simulate(), x[n].trace);
        } else {
          w[n] <- play.handle(x[n].simulate(), x[n].trace);
        }
      }
    }
  }

  override function filter(archetype:Model, t:Integer) {
    if r? && ancestor {
      /* ancestor sampling */
      auto w' <- w;
      dynamic parallel for n in 1..nparticles {
        auto x' <- clone(x[n]);
        auto r' <- clone(r!);
        w'[n] <- w'[n] + replay.handle(r', x'.simulate(t));
        ///@todo Don't assume Markov model here
      }
      b <- global.ancestor(w');
    }
    
    resample();
    parallel for n in 1..nparticles {
      if r? && n == b {
        w[n] <- replay.handle(r!, x[n].simulate(t), x[n].trace);
      } else {
        w[n] <- play.handle(x[n].simulate(t), x[n].trace);
      }
    }
    reduce();
  }

  override function resample() {
    if ess <= trigger*nparticles {
      if r? {
        (a, b) <- conditional_resample_multinomial(w, b);
      } else {
        a <- resample_multinomial(w);
      }
      dynamic parallel for n in 1..nparticles {
        if a[n] != n {
          x[n] <- clone(x[a[n]]);
        }
        w[n] <- 0.0;
      }
    } else {
      /* normalize weights to sum to nparticles */
      w <- w - lsum + log(Real(nparticles));
    }
  }

  override function read(buffer:Buffer) {
    super.read(buffer);
    ancestor <-? buffer.get("ancestor", ancestor);
  }

  override function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("ancestor", ancestor);
  }
}
