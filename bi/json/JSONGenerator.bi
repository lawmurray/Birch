hpp{{
#include <yaml.h>
}}

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
  
  hpp{{
  yaml_emitter_t emitter;
  yaml_event_t event;
  }}

  function generate(path:String, root:Value) {
    file:File <- fopen(path, WRITE);
    cpp{{
    /* initialize */
    yaml_emitter_initialize(&self->emitter);
    yaml_emitter_set_output_file(&self->emitter, file);
    
    /* start stream */
    yaml_stream_start_event_initialize(&event, YAML_UTF8_ENCODING);
    if (!yaml_emitter_emit(&self->emitter, &event)) {
      error("emission error");
    }
    
    /* output data */
    root->accept(self);
    
    /* end stream */
    yaml_stream_end_event_initialize(&event);
    if (!yaml_emitter_emit(&emitter, &event)) {
      error("emission error");
    }
    
    /* finalize */
    yaml_emitter_delete(&emitter);
    }}
  }

  function visit(value:ObjectValue) {
    cpp{{
    yaml_mapping_start_event_initialize(&event, NULL, NULL, 1,
        YAML_FLOW_MAPPING_STYLE);
    if (!yaml_emitter_emit(&emitter, &event)) {
      error("emission error");
    }
    }}    
    auto entry <- value.entries.walk();
    while entry? {
      auto e <- entry!;
      cpp{{
      yaml_scalar_event_initialize(&event, NULL, NULL,
          (yaml_char_t*)e->key.c_str(), e->key.length(), 1, 1,
          YAML_DOUBLE_QUOTED_SCALAR_STYLE);
      }}
      e.value.value.accept(this);
    }
    cpp{{
    yaml_mapping_end_event_initialize(&event);
    if (!yaml_emitter_emit(&emitter, &event)) {
      error("emission error");
    }
    }}    
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
