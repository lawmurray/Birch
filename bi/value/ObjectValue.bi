/**
 * Object value.
 */
class ObjectValue < Value {
  entries:List<Entry>;

  function accept(gen:Generator) {
    gen.visit(this);
  }

  function isObject() -> Boolean {
    return true;
  }
  
  function push(key:String, value:Value) {
    buffer:MemoryBuffer;
    buffer.value <- value;
    entry:Entry;
    entry.key <- key;
    entry.value <- buffer;
    entries.pushBack(entry);
  }
}

function ObjectValue() -> ObjectValue {
  o:ObjectValue;
  return o;
}
