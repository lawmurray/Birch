/**
 * Particle Gibbs sampler.
 */
class ParticleGibbs < ConditionalParticleFilter {  
  function start() {
    if x'? {
      /* compute the conditional distribution over the start, given the
       * steps */
      auto x <- clone<ForwardModel>(archetype!);
      auto h <- clone<EventHandler>(x'!.getHandler());
      
      h.rewind();
      h.setDiscard(true);
      x.setHandler(h);
      x.start();
      h.setDiscard(false);
      for auto t in 1..T {
        x.step();
      }

      /* simulate the conditional distribution over the start, given the
       * steps */
      x <- clone<ForwardModel>(archetype!);
      h <- x'!.getHandler();
      
      h.rewind();
      h.setDelay(false);
      x.setHandler(h);
      x.start();
      h.setDelay(true);

      /* clone to all particles, trimming the replay trace for all but the
       * reference path */
      for auto n in 1..N-1 {
        this.x[n] <- clone<ForwardModel>(x);
        this.x[n].getHandler().setReplay(nil);  // don't replay
      }
      this.x[N] <- x;  // do replay
    } else {
      super.start();
    }
  }
}
