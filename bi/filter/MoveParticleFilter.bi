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

  override function start() {
    parallel for n in 1..nparticles {
      auto play <- MoveHandler(delayed, scale);
      auto x <- MoveParticle?(this.x[n])!;
      w[n] <- w[n] + play.handle(x.m.simulate());
      w[n] <- w[n] + x.add(play.z);
    }
  }

  override function step(t:Integer) {
    parallel for n in 1..nparticles {
      auto play <- MoveHandler(delayed, scale/(t + 1.0));
      auto x <- MoveParticle?(this.x[n])!;
      w[n] <- w[n] + play.handle(x.m.simulate(t));
      w[n] <- w[n] + x.add(play.z);
    }
  }

  override function resample() {
    /* throughout, we use the property that `a[n] == n` if and only if
     * particle `n` has survived resampling */
    auto triggered <- ess <= trigger*nparticles;
    if triggered {
      a <- resample_systematic(w);
      w <- vector(0.0, nparticles);
    } else {
      /* normalize weights to sum to nparticles */
      w <- w - lsum + log(Real(nparticles));
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
      parallel for n in 1..nparticles {
        auto x <- MoveParticle?(this.x[n])!;
        for m in 1..nmoves {
          auto x' <- clone(x);
          x'.move();
          ratio:Real/* <- ratio(x', x, scale) */;          
          if log(simulate_uniform(0.0, 1.0)) <= x'.π - x.π + ratio {
            x <- x';  // accept
          }
        }
        this.x[n] <- x;
      }
    }
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    scale <-? buffer.get("scale", scale);
    nmoves <-? buffer.get("nmoves", nmoves);
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("scale", scale);
    buffer.set("nmoves", nmoves);
  }
}
