/**
 * Nil value.
 */
class NilValue < Value {
  function accept(gen:Generator) {
    gen.visit(this);
  }

  function isScalar() -> Boolean {
    return true;
  }
}

function NilValue() -> NilValue {
  o:NilValue;
  return o;
}
