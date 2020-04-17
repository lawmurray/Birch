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
   * Replay event handler, used for the reference particle.
   */
  replay:TraceHandler <- global.replay;

  override function filter(archetype:Model) {
    super.filter(archetype);
    b <- 1;
    
    /* replay event handler */
    if delayed {
      replay <- global.replayDelay;
    } else {
      replay <- global.replay;
    }
  }

  override function filter(archetype:Model, t:Integer) {
    if r? && ancestor {
      ancestorSample(t);
    }
    resample();
    step(t);
    reduce();
  }

  function ancestorSample(t:Integer) {
    auto w' <- w;
    dynamic parallel for n in 1..nparticles {
      auto x' <- clone(x[n]);
      auto r' <- clone(r!);
      w'[n] <- w'[n] + replay.handle(r', x'.simulate(t));
      ///@todo Don't assume Markov model here
    }
    b <- global.ancestor(w');
  }

  override function start() {
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

  override function step(t:Integer) {
    parallel for n in 1..nparticles {
      if r? && n == b {
        w[n] <- w[n] + replay.handle(r!, x[n].simulate(t), x[n].trace);
      } else {
        w[n] <- w[n] + play.handle(x[n].simulate(t), x[n].trace);
      }
    }
  }

  override function resample() {
    if ess <= trigger*nparticles {
      if r? {
        (a, b) <- conditional_resample_multinomial(w, b);
      } else {
        a <- resample_multinomial(w);
      }
      copy();
      w <- vector(0.0, nparticles);
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
