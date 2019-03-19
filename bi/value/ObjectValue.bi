/**
 * Object value.
 */
class ObjectValue < ArrayValue {
  keys:List<String>;

  function accept(gen:Generator) {
    gen.visit(this);
  }

  function isObject() -> Boolean {
    return true;
  }
}
