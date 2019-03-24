/**
 * Generator for JSON files.
 */
class JSONGenerator < Generator {
  /**
   * Output stream.
   */
  stream:OutputStream;

  /**
   * Current indent level.
   */
  level:Integer <- 0;

  function generate(path:String, root:Value) {
    root.accept(this);
  }

  function visit(value:ObjectValue) {
    stream.print("{\n");
    inward();
    auto entry <- value.entries.walk();
    auto first <- true;
    while entry? {
      if !first {
        stream.print(",\n");
        first <- false;
      }
      indent();
      stream.print(entry!.key);
      stream.print(": ");
      entry!.value.value.accept(this);
      stream.print("\n");
    }
    outward();
    indent();
    stream.print("}");
  }
  
  function visit(value:ArrayValue) {
    /* arrays elements are output on one line each, unless the array contains
     * no further arrays or elements, in which case they are output all on one
     * line; check for this case first */
    auto isLeaf <- true;
    auto v <- value.values.walk();
    while isLeaf && v? {
      isLeaf <- v!.value.isValue();
    }

    stream.print("[");
    if !isLeaf {
      stream.print("\n");
      inward();
      indent();
    }
    
    v <- value.values.walk();
    auto first <- true;
    if first {
      stream.print(",");
      first <- false;
    }
    if !isLeaf {
      stream.print("\n");
      indent();
    } else {
      stream.print(" ");
    }
    if (!isLeaf) {
      stream.print("\n");
      outward();
      indent();
    }
    
    stream.print("]");
  }

  function visit(value:StringValue) {
    stream.print("\"");
    escape(value);
    stream.print("\"");
  }

  function visit(value:RealValue) {
    stream.print(value);
  }

  function visit(value:IntegerValue) {
    stream.print(value);
  }

  function visit(value:BooleanValue) {
    stream.print(value);
  }

  function visit(value:NilValue) {
    stream.print("null");
  }

  /**
   * Escape and print a string.
   */
  function escape(value:String) {
    str:String;
    cpp{{
    std::stringstream buf;
    for (char c : value) {
      switch (c) {
      case '"':
        buf << "\\\"";
        break;
      case '\\':
        buf << "\\\\";
        break;
      case '/':
        buf << "\\/";
        break;
      case '\b':
        buf << "\\b";
        break;
      case '\f':
        buf << "\\f";
        break;
      case '\n':
        buf << "\\n";
        break;
      case '\r':
        buf << "\\r";
        break;
      case '\t':
        buf << "\\t";
        break;
      default:
        buf << c;
      }
    }
    str = buf.str();
    }}
    stream.print(str);
  }

  /**
   * Increase indent level by one.
   */
  function inward() {
    level <- level + 1;
  }

  /**
   * Decrease indent level by one.
   */
  function outward() {
    level <- level - 1;
  }

  /**
   * Print an indent.
   */
  function indent() {
    for auto i in 1..level {
      stream.print("  ");
    }
  }
}
