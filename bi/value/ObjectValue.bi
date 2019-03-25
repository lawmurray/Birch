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

  function getChild(name:String) -> Buffer? {
    auto entry <- entries.walk();
    while entry? {
      if entry!.name == name {
        return entry!.buffer;
      }
    }
    return nil;
  }
  
  function setChild(name:String) -> Buffer {
    buffer:MemoryBuffer;
    entry:Entry;
    entry.name <- name;
    entry.buffer <- buffer;
    entries.pushBack(entry);
    return buffer;    
  }
}

function ObjectValue() -> ObjectValue {
  o:ObjectValue;
  return o;
}
