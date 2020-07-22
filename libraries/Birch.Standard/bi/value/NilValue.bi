/**
 * Nil value.
 */
class NilValue < Value {
  function accept(writer:Writer) {
    writer.visit(this);
  }

  function isScalar() -> Boolean {
    return true;
  }
}
