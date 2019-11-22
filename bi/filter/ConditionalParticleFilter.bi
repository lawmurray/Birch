/**
 * Conditional particle filter.
 */
class ConditionalParticleFilter {
  /**
   * Number of steps.
   */
  nsteps:Integer <- 0;

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
   * Should ancestor sampling be used?
   */
  ancestor:Boolean <- false;

  fiber filter(model:Model, reference:Trace?) -> (Model[_], Real[_], Real, Real, Integer) {
    auto x <- clone<Model>(model, nparticles);  // particles
    auto w <- vector(0.0, 0);  // log-weights
    auto V <- 0.0;  // incremental log normalizing constant estimate
    auto W <- 0.0;  // cumulative log normalizing constant estimate
    auto a <- iota(1, nparticles);  // ancestor indices
    auto b <- 1;  // reference particle index
    auto ess <- 0.0;  // effective sample size
    
    /* event handlers */
    replay:TraceHandler <- global.replay;  // to replay reference particle
    play:Handler <- global.play;  // for other particles
    if delayed {
      replay <- global.redelay;
      play <- global.delay;
    }

    /* initialize and weight */
    parallel for n in 1..nparticles {
      if reference? && n == b {
        w[n] <- replay.handle(reference!, x[n].simulate(), x[n].trace);
      } else {
        w[n] <- play.handle(x[n].simulate(), x[n].trace);
      }
    }
    (ess, V) <- resample_reduce(w);
    W <- W + V;
    yield (x, w, W, ess, nparticles);
      
    auto t <- 0;
    while true {
      t <- t + 1;

      /* ancestor sampling */
      if reference? && ancestor {
        auto w' <- w;
        dynamic parallel for n in 1..nparticles {
          auto x' <- clone<Model>(x[n]);
          auto reference' <- clone<Trace>(reference!);
          w'[n] <- w'[n] + replay.handle(reference', x'.simulate(t));
          // ^ assuming Markov model here
        }

        /* simulate a new ancestor index */
        b <- global.ancestor(w');
      }
    
      /* resample */
      if ess <= trigger*nparticles {
        if reference? {
          (a, b) <- conditional_resample_multinomial(w, b);
        } else {
          a <- resample_multinomial(w);
        }
        dynamic parallel for n in 1..nparticles {
          if a[n] != n {
            x[n] <- clone<Model>(x[a[n]]);
          }
        }
      }
      
      /* propagate and weight */
      parallel for n in 1..nparticles {
        if reference? && n == b {
          w[n] <- replay.handle(reference!, x[n].simulate(t), x[n].trace);
        } else {
          w[n] <- play.handle(x[n].simulate(t), x[n].trace);
        }
      }
      (ess, V) <- resample_reduce(w);
      W <- W + V;
      yield (x, w, W, ess, nparticles);
    }
  }

  function read(buffer:Buffer) {
    nsteps <-? buffer.get("nsteps", nsteps);
    nparticles <-? buffer.get("nparticles", nparticles);
    trigger <-? buffer.get("trigger", trigger);
    delayed <-? buffer.get("delayed", delayed);
    ancestor <-? buffer.get("ancestor", ancestor);
  }

  function write(buffer:Buffer) {
    buffer.set("nsteps", nsteps);
    buffer.set("nparticles", nparticles);
    buffer.set("trigger", trigger);
    buffer.set("delayed", delayed);
    buffer.set("ancestor", ancestor);
  }
}
