/**
 * Adds ancestry sampling to a particle filter class.
 */
class WithAncestorSampling<Base> < Base {
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
        auto f <- clone<StackNode<Event>>(forward!);
        x.h.setMode(PROPOSE_IMMEDIATE);
        x.h.setRecord(false);
        x.h.trace.putForward(f, forwardCount);
        v[n] <- w[n];
        auto s <- x.t + 1;
        while v[n] > -inf && s <= T {
          v[n] <- v[n] + x.step();
          s <- s + 1;
        }
      }
      b <- ancestor(v);
      h <- x[b].h;
      h.setMode(REPLAY_DELAY);
      auto f <- clone<StackNode<Event>>(forward!);
      h.trace.putForward(f, forwardCount);
    }
    super.resample();
  }
}
