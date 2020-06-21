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

  function particle(archetype:Model) -> Particle {
    return MoveParticle(archetype);
  }

  override function filter(archetype:Model) {
    super.filter(archetype);
    naccepts <- vector(0, nparticles);
  }

  override function start() {
    parallel for n in 1..nparticles {
      auto play <- MoveHandler(delayed);
      auto x <- MoveParticle?(this.x[n])!;
      w[n] <- w[n] + play.handle(x.m.simulate());
      w[n] <- w[n] + x.add(play.z);
    }
  }

  override function step(t:Integer) {
    parallel for n in 1..nparticles {
      auto play <- MoveHandler(delayed);
      auto x <- MoveParticle?(this.x[n])!;
      w[n] <- w[n] + play.handle(x.m.simulate(t));
      w[n] <- w[n] + x.add(play.z);
    }
  }

  override function resample(t:Integer) {
    naccepts <- vector(0, nparticles);
  
    /* throughout, we use the property that `a[n] == n` if and only if
     * particle `n` has survived resampling */
    auto triggered <- ess <= trigger*nparticles;
    if triggered {
      a <- resample_systematic(w);
      w <- vector(0.0, nparticles);
    } else {
      /* normalize weights to sum to nparticles */
      w <- w - vector(lsum - log(Real(nparticles)), nparticles);
    }
    
    /* update prior for surviving particles */
    dynamic parallel for n in 1..nparticles {
      if a[n] == n {
        auto x <- MoveParticle?(this.x[n])!;
        x.prior();
      }
    }

    if triggered {
      /* calculate derivatives */
      dynamic parallel for n in 1..nparticles {
        if a[n] == n {  // if particle `n` survives, then `a[n] == n`
          auto x <- MoveParticle?(this.x[n])!;
          x.grad();
        }
      }

      /* copy particles */
      dynamic parallel for n in 1..nparticles {
        if a[n] != n {
          x[n] <- clone(x[a[n]]);
        }
      }
      
      /* move particles */
      κ:LangevinKernel;
      κ.scale <- scale/pow(t + 1.0, 3.0);
      parallel for n in 1..nparticles {
        auto x <- MoveParticle?(this.x[n])!;
        for m in 1..nmoves {
          auto x' <- clone(x);
          x'.move(κ);
          x'.grad();
          auto α <- x'.π - x.π + x'.logpdf(x, κ) - x.logpdf(x', κ);          
          if log(simulate_uniform(0.0, 1.0)) <= α {
            /* accept */
            x <- x';
            naccepts[n] <- naccepts[n] + 1;
          }
        }
        this.x[n] <- x;
      }
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
  }

  override function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("scale", scale);
    buffer.set("nmoves", nmoves);
  }
}
