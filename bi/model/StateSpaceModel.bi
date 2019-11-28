/**
 * State-space model, synonymous to `HiddenMarkovModel`.
 *
 * The Model class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Model.svg"></object>
 * </center>
 */
class StateSpaceModel<Parameter,State,Observation> =
    HiddenMarkovModel<Parameter,State,Observation>;
