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
   * Tape<Record> of the reference particle. This will have no value for the first
   * iteration of the filter. Subsequent iterations will draw a particle from
   * the previous iteration to condition the new iteration, setting this
   * variable.
   */
  r:Tape<Record>?;

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

  function particle(archetype:Model) -> Particle {
    return ConditionalParticle(archetype);
  }

  override function filter(archetype:Model) {
    super.filter(archetype);
    b <- 1;
  }

  override function filter(archetype:Model, t:Integer) {
    if r? && ancestor {
      ancestorSample(t);
    }
    resample(t);
    step(t);
    reduce();
  }

  function ancestorSample(t:Integer) {
    auto play <- PlayHandler(delayed);
    auto w' <- w;
    dynamic parallel for n in 1..nparticles {
      auto x' <- clone(x[n]);
      auto r' <- clone(r!);
      w'[n] <- w'[n] + play.handle(r', x'.m.simulate(t));
      ///@todo Don't assume Markov model here
    }
    b <- global.ancestor(w');
  }

  override function start() {
    auto play <- PlayHandler(delayed);
    if !alreadyInitialized {
      parallel for n in 1..nparticles {
        auto x <- ConditionalParticle?(this.x[n])!;
        if r? && n == b {
          w[n] <- play.handle(r!, x.m.simulate(), x.trace);
        } else {
          w[n] <- play.handle(x.m.simulate(), x.trace);
        }
      }
    }
  }

  override function step(t:Integer) {
    auto play <- PlayHandler(delayed);
    parallel for n in 1..nparticles {
        auto x <- ConditionalParticle?(this.x[n])!;
      if r? && n == b {
        w[n] <- w[n] + play.handle(r!, x.m.simulate(t), x.trace);
      } else {
        w[n] <- w[n] + play.handle(x.m.simulate(t), x.trace);
      }
    }
  }

  override function resample(t:Integer) {
    if ess <= trigger*nparticles {
      if r? {
        (a, b) <- conditional_resample_multinomial(w, b);
      } else {
        a <- resample_multinomial(w);
      }
      w <- vector(0.0, nparticles);
      dynamic parallel for n in 1..nparticles {
        if a[n] != n {
          x[n] <- clone(x[a[n]]);
        }
      }
    } else {
      /* normalize weights to sum to nparticles */
      w <- w - vector(lsum - log(Real(nparticles)), nparticles);
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
