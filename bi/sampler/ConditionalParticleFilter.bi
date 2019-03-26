/**
 * Conditional particle filter.
 */
class ConditionalParticleFilter < ParticleFilter {
  function start() {
    if x'? {
      x'!.getHandler().replay();
      x[N].setHandler(x'!.getHandler());
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
      /* move the replay event handler out of the reference particle so that
       * it won't be copied to other offspring */
      x[N].getHandler().rebase(DelayHandler());
    }
    super.copy();
    if x'? {
      /* move the replay event handler back in to the reference particle */
      x[N].getHandler().rebase(x'!.getHandler());
    }
  }
  
  function setArchetype(a:Model) {
    super.setArchetype(a);
    archetype!.setHandler(TraceHandler(archetype!.getHandler()));
  }
}
