/**
 * Particle Gibbs sampler.
 */
class ParticleGibbs < ConditionalParticleFilter {  
  function start() {
    /* perform a Gibbs move on the start of the reference path */
    auto x <- clone<ForwardModel>(archetype!);
    auto h <- clone<EventHandler>(x'!.getHandler());
    x.setHandler(h);
    h.rewind();
    
    /* replay the start, but discard the trace so as to re-establish the
     * prior distribution over the start */
    h.setDiscard(true);
    x.start();
    h.setDiscard(false);
    
    /* replay the steps, so as to compute the conditional distribution of
     * the start given the steps */
    for auto t in 1..T {
      x.step();
    }

    /* rewind again and replay the start, this time simulating immediately,
     * from the conditional distribution of the start given the steps */
    x <- clone<ForwardModel>(archetype!);
    x.setHandler(h);
    h.rewind();
    h.setDelay(false);
    x.start();
    h.setDelay(true);
    
    /* now clone that to all particles */
    for auto n in 1..N {
      this.x[n] <- clone<ForwardModel>(x);
    }
  }
}
