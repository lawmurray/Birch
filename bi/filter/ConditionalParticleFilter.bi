/**
 * Conditional particle filter.
 */
class ConditionalParticleFilter {
  /**
   * Model.
   */
  model:ForwardModel;

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

  fiber filter(model:ForwardModel, reference:Trace?) -> (ForwardModel[_],
      Real[_], Trace[_], Real, Real) {
    auto x <- clone<ForwardModel>(model, nparticles);  // particles
    auto w <- vector(0.0, 0);  // log-weights
    auto a <- iota(1, nparticles);  // ancestor indices
    auto b <- 1;  // reference particle index
    auto ess <- 0.0;  // effective sample size
    auto levidence <- 0.0;  // incremental log-evidence
    r:Trace[nparticles];  // recorded traces
    
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
        w[n] <- replay.handle(reference!, x[n].simulate(), r[n]);
      } else {
        w[n] <- play.handle(x[n].simulate(), r[n]);
      }
    }
    (ess, levidence) <- resample_reduce(w);
    yield (x, w, r, ess, levidence);
      
    auto t <- 0;
    while true {
      t <- t + 1;

      /* ancestor sampling */
      if reference? && ancestor {
        auto w' <- w;
        dynamic parallel for n in 1..nparticles {
          auto x' <- clone<ForwardModel>(x[n]);
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
            x[n] <- clone<ForwardModel>(x[a[n]]);
            r[n] <- clone<Trace>(r[a[n]]);
          }
        }
      }
      
      /* propagate and weight */
      parallel for n in 1..nparticles {
        if reference? && n == b {
          w[n] <- replay.handle(reference!, x[n].simulate(t), r[n]);
        } else {
          w[n] <- play.handle(x[n].simulate(t), r[n]);
        }
      }
      yield (x, w, r, ess, levidence);
    }
  }

  function read(buffer:Buffer) {
    nparticles <-? buffer.get("nparticles", nparticles);
    trigger <-? buffer.get("trigger", trigger);
    delayed <-? buffer.get("delayed", delayed);
    ancestor <-? buffer.get("ancestor", ancestor);
  }

  function write(buffer:Buffer) {
    buffer.set("nparticles", nparticles);
    buffer.set("trigger", trigger);
    buffer.set("delayed", delayed);
    buffer.set("ancestor", ancestor);
  }
}
