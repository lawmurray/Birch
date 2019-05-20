class HMMWithProposal<Parameter,State,Observation> < HiddenMarkovModel<Parameter,State,Observation> {

  accept:Boolean;

  function propose(x':HMMWithProposal<Parameter,State,Observation>) -> (Real, Real) {
  }
  
  function write(buf:Buffer) {
    super.write(buf);
    buf.set("accept", accept);
  }
}
