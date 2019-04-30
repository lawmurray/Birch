/**
 * Conditional particle filter. This behaves as per ParticleFilter for the
 * first sample. For subsequent samples it conditions on a particle drawn
 * from the previous iteration.
 */
class ConditionalParticleFilter < ParticleFilter {
  /**
   * Current index of the reference trajectory.
   */
  b:Integer <- 1;
  
  function initialize() {
    super.initialize();

    /* switch on recording for all particles */
    parallel for auto n in 1..N {
      x[n].getHandler().setRecord(true);
    }
  
    if x'? {
      /* there is a reference particle, switch on replay for it */
      auto h <- clone<EventHandler>(x'!.getHandler());
      h.rewind();
      h.setMode(REPLAY_DELAY);
      x[1].setHandler(h);
      b <- 1;
    }
  }

  function resample() {
    if ess.back() <= trigger*N {
      /* temporarily remove the replay trace from the reference particle to
       * avoid copying it around */
      auto forward <- x[b].getHandler().trace.forward;
      x[b].getHandler().trace.forward <- nil;
      x[b].getHandler().setMode(PLAY_DELAY);
      
      /* resample */
      if x'? {
        (a, o, b) <- multinomial_conditional_resample(w, b);
      } else {
        (a, o) <- multinomial_resample(w);
      }
      w <- vector(0.0, N);

      /* copy particles */
      auto x0 <- x;
      parallel for auto n in 1..N {
        if o[a[n]] == 1 {
          x[n] <- x0[a[n]];  // avoid the clone overhead
        } else {
          x[n] <- clone<ForwardModel>(x0[a[n]]);
        }
      }
      
      /* restore replay trace to new reference particle */
      if x'? {
        x[b].getHandler().setMode(REPLAY_DELAY);
        x[b].getHandler().trace.forward <- forward;
      }
    } else {
      a <- iota(1, N);
      o <- vector(1, N);
    }
  }
}
