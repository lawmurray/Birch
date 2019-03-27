/**
 * Conditional particle filter.
 */
class ConditionalParticleFilter < ParticleFilter {
  function initialize() {
    super.initialize();
    if x'? {
      auto h <- x'!.getHandler();
      h.replay();
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
      /* temporarily move the replay event handler out of the reference
       * particle so that be copied to other offspring */
      auto h <- x[N].getHandler();
      //h.rebase(DelayedHandler());
      super.copy();
      //h.rebase(x'!.getHandler());
    } else {
      super.copy();
    }
  }
  
  function setArchetype(a:Model) {
    super.setArchetype(a);
    //archetype!.setHandler(TraceHandler(archetype!.getHandler()));
  }
}
