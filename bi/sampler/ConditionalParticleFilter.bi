/**
 * Conditional particle filter.
 */
class ConditionalParticleFilter < ParticleFilter {
  function start() {
    super.start();
    
    /* install a replay handler for the reference particle */
    if x'? {
      auto h <- clone<EventHandler>(x'!.getHandler());
      h.rewind();
      x[N].setHandler(h);
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
      auto replay <- x[N].getHandler().takeReplay();
      super.copy();
      x[N].getHandler().setReplay(replay);
    } else {
      super.copy();
    }
  }
  
  function setArchetype(a:Model) {
    super.setArchetype(a);
    h:TraceHandler;
    archetype!.setHandler(h);
  }
}
