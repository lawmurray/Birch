/**
 * Conditional particle filter. This behaves as per ParticleFilter for the
 * first sample. For subsequent samples it conditions on a particle drawn
 * from the previous iteration.
 */
class ConditionalParticleFilter < ParticleFilter {
  /**
   * Event handler of the reference trajectory.
   */
  h':EventHandler?;
  
  function initialize() {
    super.initialize();
      
    if h'? {
      /* there is a reference particle, switch on replay for it */
      x[b].h.setMode(REPLAY_DELAY);
      x[b].h.trace.forwardCount <- h'!.trace.forwardCount;
      x[b].h.trace.forward <- h'!.trace.takeForward();
    }
    
    /* switch on recording for all particles */
    for auto n in 1..N {
      x[n].h.setRecord(true);
    }
  }

  function resample() {
    if ess.back() <= trigger*N {
      /* temporarily remove the replay trace from the reference particle to
       * avoid copying it around */
      auto forwardCount <- x[b].h.trace.forwardCount;
      auto forward <- x[b].h.trace.takeForward();
      x[b].h.setMode(PLAY_DELAY);
      
      /* resample */
      if h'? {
        (a, b) <- multinomial_conditional_resample(w, b);
      } else {
        a <- multinomial_resample(w);
      }
      w <- vector(0.0, N);

      /* copy particles */
      dynamic parallel for auto n in 1..N {
        if a[n] == n {
          // avoid clone overhead
        } else {
          x[n] <- clone<ForwardModel>(x[a[n]]);
        }
      }
      
      /* restore replay trace to new reference particle */
      if h'? {
        x[b].h.setMode(REPLAY_DELAY);
        x[b].h.trace.putForward(forward, forwardCount);
      }
    } else {
      a <- iota(1, N);
    }
  }

  function finish() {
    super.finish();
    h' <- x[b].h;
    h'!.rewind();
  }
}
