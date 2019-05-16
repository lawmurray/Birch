/**
 * Particle Gibbs with ancestor sampling.
 */
class ParticleGibbsWithAncestorSampling < ParticleGibbs {
  function resample() {
    if h'? {
      /* get the reference trace */
      auto h <- x[b].h;
      h.setMode(PLAY_DELAY);
      auto forwardCount <- h.trace.forwardCount;
      auto forward <- h.trace.takeForward();

      /* simulate a new ancestor index; for each particle, take one step
       * forward, proposing values given from the reference trace */
      v:Real[N];
      parallel for auto n in 1..N {
        auto x <- clone<ForwardModel>(this.x[n]);
        x.h.setMode(PROPOSE_IMMEDIATE);
        x.h.trace.forward <- clone<StackNode<Event>>(forward!);
        v[n] <- w[n] + x.step();
      }
      b <- ancestor(v);
      h <- x[b].h;
      h.setMode(REPLAY_DELAY);
      h.trace.putForward(forward, forwardCount);
    }
    super.resample();
  }
}
