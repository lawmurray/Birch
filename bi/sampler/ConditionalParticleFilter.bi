/**
 * Conditional particle filter.
 */
class ConditionalParticleFilter < ParticleFilter {
  function start() {
    /* turn on recording for all particles */
    parallel for auto n in 1..N {
      x[n].getHandler().setRecord(true);
    }
  
    /* turn on replay for the reference particle */
    if x'? {
      auto h <- x'!.getHandler();
      h.rewind();
      h.setMode(REPLAY_DELAY);
      x[N].setHandler(h);
    }
    
    super.start();
  }

  function resample() {
    if ess.back() <= trigger*N {
      if x'? {
        (a, o) <- multinomial_conditional_resample(w);
      } else {
        (a, o) <- multinomial_resample(w);
      }
      w <- vector(0.0, N);
    } else {
      a <- iota(1, N);
      o <- vector(1, N);
    }
  }
  
  function step() {      
    if x'? {
      /* step all but the reference particle; temporarily take the replay
       * trace out of the reference particle so as not to copy it into
       * offspring */
      auto x0 <- x;
      auto forward <- x[N].getHandler().trace.forward;
      x[N].getHandler().trace.forward <- nil;
      x[N].getHandler().setMode(PLAY_DELAY);
      ///@todo create an interface to remove the reference trace rather
      ///than playing with Queue internals      
      parallel for auto n in 1..N-1 {
        if o[a[n]] == 1 {
          x[n] <- x0[a[n]];  // avoid the clone overhead
        } else {
          x[n] <- clone<ForwardModel>(x0[a[n]]);
        }
        w[n] <- w[n] + x[n].play();
      }
      
      /* step the reference particle */
      x[N].getHandler().setMode(REPLAY_DELAY);
      x[N].getHandler().trace.forward <- forward;
      if o[N] == 1 {
        x[N] <- x0[N];  // avoid the clone overhead
      } else {
        x[N] <- clone<ForwardModel>(x0[N]);
      }
      w[N] <- w[N] + x[N].play();
    } else {
      super.step();
    }
  }
}
