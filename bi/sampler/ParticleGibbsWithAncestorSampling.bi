/**
 * Particle Gibbs with ancestor sampling.
 */
class ParticleGibbsWithAncestorSampling < ParticleGibbs {
  function resample() {
    if h'? {
      /* get the reference trace */
      auto forward <- x[b].getHandler().trace.forward;
      x[b].getHandler().trace.forward <- nil;
      x[b].getHandler().setMode(PLAY_DELAY);

      /* simulate a new ancestor index; for each particle, take one step
       * forward, proposing values given from the reference trace */
      v:Real[N];
      parallel for auto n in 1..N {
        auto x <- clone<ForwardModel>(this.x[n]);
        x.getHandler().setMode(PROPOSE_IMMEDIATE);
        x.getHandler().trace.forward <- clone<StackNode<Event>>(forward!);
        v[n] <- w[n] + x.step();
      }
      b <- ancestor(v);
      x[b].getHandler().trace.forward <- forward;
      x[b].getHandler().setMode(REPLAY_DELAY);
    }
    super.resample();
  }
}
