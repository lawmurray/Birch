/**
 * Particle Gibbs sampler.
 */
class ParticleGibbs < ConditionalParticleFilter {  
  function start() {
    if x'? {
      /* compute the conditional distribution over the start, given the
       * steps */
      auto x <- clone<ForwardModel>(archetype!);
      auto h <- x'!.getHandler();
      
      h.rewind();
      h.setMode(PLAY_DELAY);
      x.setHandler(h);
      x.start();
      h.setMode(REPLAY_DELAY);
      for auto t in 1..T {
        x.step();
      }

      /* simulate the conditional distribution over the start, given the
       * steps */
      x <- clone<ForwardModel>(archetype!);
      h <- clone<EventHandler>(x'!.getHandler());
      
      h.rewind();
      h.setMode(PLAY_IMMEDIATE);
      x.setHandler(h);
      x.start();
      h.setMode(REPLAY_DELAY);

      /* clone to all particles */
      auto forward <- x.getHandler().trace.forward;
      x.getHandler().trace.forward <- nil;
      for auto n in 1..N {
        this.x[n] <- clone<ForwardModel>(x);
      }
      this.x[N].getHandler().trace.forward <- forward;
    } else {
      super.start();
    }
  }
}
