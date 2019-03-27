/**
 * Conditional particle filter.
 */
class ConditionalParticleFilter < ParticleFilter {
  function initialize() {
    super.initialize();
    if x'? {
      auto h <- clone<Handler>(x'!.getHandler());
      x[N].setHandler(h);
      h.rewind();
    }
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
      auto h <- x[N].getHandler().takeReplay();
      super.copy();
      x[N].getHandler().setReplay(h);
    } else {
      super.copy();
    }
  }
  
  function setArchetype(a:Model) {
    super.setArchetype(a);
    h:TraceHandler<DelayedReplayHandler>;
    archetype!.setHandler(h);
  }
}
