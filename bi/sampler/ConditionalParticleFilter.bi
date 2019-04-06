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
      h.setMode(REPLAY_UPDATE);
      x[N].setHandler(h);
    }
    super.start();
  }

  function resample() {
    if x'? {
      a <- multinomial_conditional_ancestors(w);
    } else {
      a <- multinomial_ancestors(w);
    }
  }
  
  function copy() {
    if x'? {
      /* temporarily take the replay trace out of the reference particle so
       * as not to copy it into offspring */
      x[N].getHandler().setMode(PLAY_DELAY);
      auto replay <- x[N].getHandler().takeReplay();
      super.copy();
      x[N].getHandler().setMode(REPLAY_UPDATE);
      x[N].getHandler().setReplay(replay);
    } else {
      super.copy();
    }
  }
}
