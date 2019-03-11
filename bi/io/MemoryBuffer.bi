/**
 * Buffer in memory.
 */
class MemoryBuffer < Buffer {
  hpp{{
  libubjpp::value root;
  }}
  
  function get() -> Buffer {
    buffer:MemoryBufferAuxiliary;
    cpp{{
    buffer->group = &self->root;
    }}
    return buffer;
  }

  function getChild(name:String) -> Buffer? {
    exists:Boolean <- false;
    cpp{{
    auto child = root.get(name);
    exists = static_cast<bool>(child);
    }}
    if (exists) {
      buffer:MemoryBufferAuxiliary;
      cpp{{
      buffer->group = &child.get();
      }}
      return buffer;
    } else {
      return nil;
    }
  }

  function set() -> Buffer {
    buffer:MemoryBufferAuxiliary;
    cpp{{
    buffer->group = &self->root.set();
    }}
    return buffer;
  }

  function setChild(name:String) -> Buffer {
    buffer:MemoryBufferAuxiliary;
    cpp{{
    buffer->group = &self->root.set(name);
    }}
    return buffer;
  }
}
