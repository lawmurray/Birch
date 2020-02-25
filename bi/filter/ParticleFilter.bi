/**
 * Particle filter.
 *
 * The ParticleFilter class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Filter.svg"></object>
 * </center>
 */
class ParticleFilter {
  /**
   * Number of steps. If this has no value, the model will be required to
   * suggest an appropriate value.
   */
  nsteps:Integer?;

  /**
   * Number of additional forecast steps per step.
   */
  nforecasts:Integer <- 0;
  
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
   * Should ancestor sampling be used for conditional filter?
   */
  ancestor:Boolean <- false;

  /**
   * Filter.
   *
   * - model: The model.
   *
   * Yields: a tuple giving, in order:
   *   - particle states,
   *   - particle log weights,
   *   - log normalizing constant estimate,
   *   - effective sample size,
   *   - total number of propagations used to obtain these, which may include
   *     rejected particles.
   */
  fiber filter(model:Model) -> (Model[_], Real[_], Real, Real, Integer) {
    auto x <- clone<Model>(model, nparticles);  // particles
    auto w <- vector(0.0, nparticles);  // log weights
    auto ess <- 0.0;  // effective sample size
    auto S <- 0.0;  // logarithm of the sum of weights
    auto W <- 0.0;  // cumulative log normalizing constant estimate
    
    /* number of steps */
    if !nsteps? {
      nsteps <- model.size();
    }
    
    /* event handler */
    h:Handler <- play;
    if delayed {
      h <- global.playDelay;
    }

    /* initialize and weight */
    parallel for n in 1..nparticles {
      w[n] <- h.handle(x[n].simulate());
    }
    (ess, S) <- resample_reduce(w);
    W <- W + S - log(Real(nparticles));
    yield (x, w, W, ess, nparticles);
    
    for t in 1..nsteps! {
      /* resample */
      if ess <= trigger*nparticles {
        auto a <- resample_systematic(w);
        dynamic parallel for n in 1..nparticles {
          if a[n] != n {
            x[n] <- clone<Model>(x[a[n]]);
          }
          w[n] <- 0.0;
        }
      } else {
        /* normalize weights to sum to nparticles */
        w <- w - (S - log(Real(nparticles)));
      }
      
      /* propagate and weight */
      parallel for n in 1..nparticles {
        w[n] <- w[n] + h.handle(x[n].simulate(t));
      }
      (ess, S) <- resample_reduce(w);
      W <- W + S - log(Real(nparticles));
      yield (x, w, W, ess, nparticles);
    }
  }

  fiber filter(model:Model, reference:Trace?, alreadyInitialized:Boolean) ->
      (Model[_], Real[_], Real, Real, Integer) {
    auto x <- clone<Model>(model, nparticles);  // particles
    auto w <- vector(0.0, nparticles);  // log-weights
    auto ess <- 1.0*nparticles;  // effective sample size
    auto S <- 0.0;  // logarithm of the sum of weights
    auto W <- 0.0;  // cumulative log normalizing constant estimate
    auto a <- iota(1, nparticles);  // ancestor indices
    auto b <- 1;  // reference particle index

    /* number of steps */
    if !nsteps? {
      nsteps <- model.size();
    }
    
    /* event handlers */
    replay:TraceHandler <- global.replay;  // to replay reference particle
    play:Handler <- global.play;  // for other particles
    if delayed {
      play <- global.playDelay;
      replay <- global.replayDelay;
    }

    /* initialize and weight */
    if !alreadyInitialized {
      parallel for n in 1..nparticles {
        if reference? && n == b {
          w[n] <- replay.handle(reference!, x[n].simulate(), x[n].trace);
        } else {
          w[n] <- play.handle(x[n].simulate(), x[n].trace);
        }
      }
      (ess, S) <- resample_reduce(w);
      W <- W + S - log(Real(nparticles));
      yield (x, w, W, ess, nparticles);
    }
    
    for t in 1..nsteps! {
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
          w[n] <- 0.0;
        }
      } else {
        /* normalize weights to sum to nparticles */
        w <- w - (S - log(Real(nparticles)));
      }
      
      /* propagate and weight */
      parallel for n in 1..nparticles {
        if reference? && n == b {
          w[n] <- replay.handle(reference!, x[n].simulate(t), x[n].trace);
        } else {
          w[n] <- play.handle(x[n].simulate(t), x[n].trace);
        }
      }
      (ess, S) <- resample_reduce(w);
      W <- W + S - log(Real(nparticles));
      yield (x, w, W, ess, nparticles);
    }
  }

  /**
   * Forecast.
   *
   * - t: Time step.
   * - x: Starting states.
   * - w: Starting log weights.
   *
   * Yields: a tuple giving, in order:
   *   - particle states,
   *   - particle log weights.
   */
  fiber forecast(t:Integer, x:Model[_], w:Real[_]) -> (Model[_],
      Real[_]) {
    assert length(x) == nparticles;

    auto x' <- x;
    auto w' <- w;

    /* resample */
    ess:Real;
    S:Real;
    (ess, S) <- resample_reduce(w);
    if ess <= trigger*nparticles {
      auto a <- resample_systematic(w);
      dynamic parallel for n in 1..nparticles {
        x'[n] <- clone<Model>(x[a[n]]);
        w'[n] <- 0.0;
      }
    } else {
      dynamic parallel for n in 1..nparticles {
        x'[n] <- clone<Model>(x[n]);
      }
    }

    /* forecast */
    for s in 1..nforecasts {
      parallel for n in 1..nparticles {
        w'[n] <- w'[n] + play.handle(x'[n].forecast(t + s));
      }
      yield (x', w');
    }
  }

  function read(buffer:Buffer) {
    nsteps <-? buffer.get("nsteps", nsteps);
    nforecasts <-? buffer.get("nforecasts", nforecasts);
    nparticles <-? buffer.get("nparticles", nparticles);
    trigger <-? buffer.get("trigger", trigger);
    delayed <-? buffer.get("delayed", delayed);
    ancestor <-? buffer.get("ancestor", ancestor);
  }

  function write(buffer:Buffer) {
    if nsteps? {
      buffer.set("nsteps", nsteps!);
    }
    buffer.set("nforecasts", nforecasts);
    buffer.set("nparticles", nparticles);
    buffer.set("trigger", trigger);
    buffer.set("delayed", delayed);
    buffer.set("ancestor", ancestor);
  }
}
