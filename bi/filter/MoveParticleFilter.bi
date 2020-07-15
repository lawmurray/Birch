/**
 * Resample-move particle filter.
 *
 * The ParticleFilter class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Filter.svg"></object>
 * </center>
 */
class MoveParticleFilter < ParticleFilter {
  /**
   * Number of moves accepted for each particle.
   */
  naccepts:Integer[_];

  /**
   * Scale of moves.
   */
  scale:Real <- 0.1;
  
  /**
   * Number of moves at each step.
   */
  nmoves:Integer <- 1;
  
  /**
   * Number of lag steps for each move.
   */
  nlags:Integer <- 1;

  function particle(archetype:Model) -> Particle {
    return MoveParticle(archetype);
  }

  override function filter(t:Integer) {
    resample(t);
    move(t);
    step(t);
    reduce();
  }

  override function start() {
    parallel for n in 1..nparticles {
      auto play <- MoveHandler(delayed);
      auto x <- MoveParticle?(this.x[n])!;
      w[n] <- w[n] + play.handle(x.m.simulate());
      w[n] <- w[n] + x.augment(0, play.z);
      while x.size() > nlags {
        x.truncate();
      }
    }
  }

  override function step(t:Integer) {
    parallel for n in 1..nparticles {
      auto play <- MoveHandler(delayed);
      auto x <- MoveParticle?(this.x[n])!;
      w[n] <- w[n] + play.handle(x.m.simulate(t));
      w[n] <- w[n] + x.augment(t, play.z);
      while x.size() > nlags {
        x.truncate();
      }
    }
  }

  function move(t:Integer) {
    naccepts <- vector(0, nparticles);
    if nlags > 0 && nmoves > 0 {
      κ:LangevinKernel;
      κ.scale <- scale/pow(t, 3);
      parallel for n in 1..nparticles {
        auto x <- MoveParticle?(clone(this.x[n]))!;
        x.grad(t - nlags);
        for m in 1..nmoves {
          auto x' <- clone(x);
          x'.move(t - nlags, κ);
          x'.grad(t - nlags);
          auto α <- x'.π - x.π + x'.logpdf(x, κ) - x.logpdf(x', κ);
          if log(simulate_uniform(0.0, 1.0)) <= α {  // accept?
            x <- x';
            naccepts[n] <- naccepts[n] + 1;
          }
        }
        this.x[n] <- x;
      }
      collect();
    }
  }

  override function reduce() {
    super.reduce();
    raccept <- Real(sum(naccepts))/(nparticles*nmoves);
  }

  override function read(buffer:Buffer) {
    super.read(buffer);
    scale <-? buffer.get("scale", scale);
    nmoves <-? buffer.get("nmoves", nmoves);
    nlags <-? buffer.get("nlags", nlags);
  }

  override function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("scale", scale);
    buffer.set("nmoves", nmoves);
    buffer.set("nlags", nlags);
  }
}
