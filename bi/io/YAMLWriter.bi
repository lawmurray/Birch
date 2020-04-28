hpp{{
#include <yaml.h>
}}

/**
 * Writer for YAML files.
 */
class YAMLWriter < Writer {
  /**
   * The file.
   */
  file:File;

  hpp{{
  yaml_emitter_t emitter;
  yaml_event_t event;
  }}
  
  function open(path:String) {
    file <- fopen(path, WRITE);
    cpp{{
    yaml_emitter_initialize(&self_()->emitter);
    yaml_emitter_set_unicode(&self_()->emitter, 1);
    yaml_emitter_set_output_file(&self_()->emitter, self_()->file);
    yaml_stream_start_event_initialize(&self_()->event, YAML_UTF8_ENCODING);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    yaml_document_start_event_initialize(&self_()->event, NULL, NULL, NULL, 1);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    }}
  }
  
  function print(buffer:MemoryBuffer) {
    buffer.value.accept(this);
  }
  
  function flush() {
    cpp{{
    yaml_emitter_flush(&self_()->emitter);
    }}
    fflush(file);
  }

  function close() {
    cpp{{
    yaml_document_end_event_initialize(&self_()->event, 1);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    yaml_stream_end_event_initialize(&self_()->event);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    yaml_emitter_delete(&self_()->emitter);
    }}
    fclose(file);
  }

  function visit(value:ObjectValue) {
    startMapping();
    auto entry <- value.entries.walk();
    while entry? {
      auto e <- entry!;
      scalar(e.name);
      e.buffer.value.accept(this);
    }
    endMapping();
  }
  
  function visit(value:ArrayValue) {
    startSequence();
    auto element <- value.buffers.walk();
    while element? {
      element!.value.accept(this);
    }
    endSequence();
  }

  function visit(value:StringValue) {
    scalar(value.value);
  }

  function visit(value:RealValue) {
    scalar(value.value);
  }

  function visit(value:IntegerValue) {
    scalar(value.value);
  }

  function visit(value:BooleanValue) {
    scalar(value.value);
  }

  function visit(value:NilValue) {
    null();
  }
  
  function visit(value:BooleanVectorValue) {
    startSequence();
    auto v <- value.value;
    for i in 1..length(v) {
      scalar(v[i]);
    }
    endSequence();
  }

  function visit(value:IntegerVectorValue) {
    startSequence();
    auto v <- value.value;
    for i in 1..length(v) {
      scalar(v[i]);
    }
    endSequence();
  }
  
  function visit(value:RealVectorValue) {
    startSequence();
    auto v <- value.value;
    for i in 1..length(v) {
      scalar(v[i]);
    }
    endSequence();
  }
  
  function visit(value:BooleanMatrixValue) {
    auto v <- value.value;
    auto m <- rows(v);
    auto n <- columns(v);
    if m > 0 {
      startSequence();
      for i in 1..m {
        if n > 0 {
          startSequence();
          for j in 1..n {
            scalar(v[i,j]);
          }
          endSequence();
        }
      }
      endSequence();
    } else {
      null();
    }
  }
  
  function visit(value:IntegerMatrixValue) {
    auto v <- value.value;
    auto m <- rows(v);
    auto n <- columns(v);
    if m > 0 {
      startSequence();
      for i in 1..m {
        if n > 0 {
          startSequence();
          for j in 1..n {
            scalar(v[i,j]);
          }
          endSequence();
        }
      }
      endSequence();
    } else {
      null();
    }
  }
  
  function visit(value:RealMatrixValue) {
    auto v <- value.value;
    auto m <- rows(v);
    auto n <- columns(v);
    if m > 0 {
      startSequence();
      for i in 1..m {
        if n > 0 {
          startSequence();
          for j in 1..n {
            scalar(v[i,j]);
          }
          endSequence();
        }
      }
      endSequence();
    } else {
      null();
    }
  }
  
  function startMapping() {
    cpp{{
    yaml_mapping_start_event_initialize(&self_()->event, NULL, NULL, 1,
        YAML_ANY_MAPPING_STYLE);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    }}
  }
  
  function endMapping() {
    cpp{{
    yaml_mapping_end_event_initialize(&self_()->event);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    }}
  }
  
  function startSequence() {
    cpp{{
    yaml_sequence_start_event_initialize(&self_()->event, NULL, NULL, 1,
        YAML_ANY_SEQUENCE_STYLE);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    }}
  }
  
  function endSequence() {
    cpp{{
    yaml_sequence_end_event_initialize(&self_()->event);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    }}    
  }
  
  function scalar(x:Boolean) {
    auto value <- String(x);
    cpp{{
    yaml_scalar_event_initialize(&self_()->event, NULL, NULL,
        (yaml_char_t*)value.c_str(), value.length(), 1, 1,
        YAML_ANY_SCALAR_STYLE);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    }}
  }

  function scalar(x:Integer) {
    auto value <- String(x);
    cpp{{
    yaml_scalar_event_initialize(&self_()->event, NULL, NULL,
        (yaml_char_t*)value.c_str(), value.length(), 1, 1,
        YAML_ANY_SCALAR_STYLE);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    }}
  }

  function scalar(x:Real) {
    auto value <- String(x);
    cpp{{
    yaml_scalar_event_initialize(&self_()->event, NULL, NULL,
        (yaml_char_t*)value.c_str(), value.length(), 1, 1,
        YAML_ANY_SCALAR_STYLE);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    }}
  }

  function scalar(value:String) {
    cpp{{
    yaml_scalar_event_initialize(&self_()->event, NULL, NULL,
        (yaml_char_t*)value.c_str(), value.length(), 1, 1,
        YAML_ANY_SCALAR_STYLE);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    }}
  }

  function null() {
    cpp{{
    yaml_scalar_event_initialize(&self_()->event, NULL, NULL,
        (yaml_char_t*)"null", 4, 1, 1,
        YAML_ANY_SCALAR_STYLE);
    yaml_emitter_emit(&self_()->emitter, &self_()->event);
    }}
  }
}
