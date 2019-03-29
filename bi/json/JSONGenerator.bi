hpp{{
#include <yaml.h>
}}

/**
 * Generator for JSON files.
 */
class JSONGenerator < Generator {  
  hpp{{
  yaml_emitter_t emitter;
  yaml_event_t event;
  }}

  function generate(path:String, buffer:MemoryBuffer) {
    auto file <- fopen(path, WRITE);
    cpp{{
    yaml_emitter_initialize(&self->emitter);
    yaml_emitter_set_output_file(&self->emitter, file);
    yaml_stream_start_event_initialize(&self->event, YAML_UTF8_ENCODING);
    yaml_emitter_emit(&self->emitter, &self->event);
    yaml_document_start_event_initialize(&self->event, NULL, NULL, NULL, 1);
    yaml_emitter_emit(&self->emitter, &self->event);
    }}
    buffer.value.accept(this);
    cpp{{
    yaml_document_end_event_initialize(&self->event, 1);
    yaml_emitter_emit(&self->emitter, &self->event);
    yaml_stream_end_event_initialize(&self->event);
    yaml_emitter_emit(&self->emitter, &self->event);
    yaml_emitter_delete(&self->emitter);
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
    scalar("null");
  }
  
  function visit(value:BooleanVectorValue) {
    startSequence();
    auto v <- value.value;
    for auto i in 1..length(v) {
      scalar(v[i]);
    }
    endSequence();
  }

  function visit(value:IntegerVectorValue) {
    startSequence();
    auto v <- value.value;
    for auto i in 1..length(v) {
      scalar(v[i]);
    }
    endSequence();
  }
  
  function visit(value:RealVectorValue) {
    startSequence();
    auto v <- value.value;
    for auto i in 1..length(v) {
      scalar(v[i]);
    }
    endSequence();
  }
  
  function visit(value:BooleanMatrixValue) {
    auto v <- value.value;
    auto m <- rows(value);
    auto n <- columns(value);
    if m > 0 {
      startSequence();
      for auto i in 1..m {
        if n > 0 {
          startSequence();
          for auto j in 1..n {
            scalar(v[i,j]);
          }
          endSequence();
        }
      }
      endSequence();
    } else {
      scalar("null");
    }
  }
  
  function visit(value:IntegerMatrixValue) {
    auto v <- value.value;
    auto m <- rows(value);
    auto n <- columns(value);
    if m > 0 {
      startSequence();
      for auto i in 1..m {
        if n > 0 {
          startSequence();
          for auto j in 1..n {
            scalar(v[i,j]);
          }
          endSequence();
        }
      }
      endSequence();
    } else {
      scalar("null");
    }
  }
  
  function visit(value:RealMatrixValue) {
    auto v <- value.value;
    auto m <- rows(value);
    auto n <- columns(value);
    if m > 0 {
      startSequence();
      for auto i in 1..m {
        if n > 0 {
          startSequence();
          for auto j in 1..n {
            scalar(v[i,j]);
          }
          endSequence();
        }
      }
      endSequence();
    } else {
      scalar("null");
    }
  }
  
  function startMapping() {
    cpp{{
    yaml_mapping_start_event_initialize(&self->event, NULL, NULL, 1,
        YAML_FLOW_MAPPING_STYLE);
    yaml_emitter_emit(&self->emitter, &self->event);
    }}
  }
  
  function endMapping() {
    cpp{{
    yaml_mapping_end_event_initialize(&self->event);
    yaml_emitter_emit(&self->emitter, &self->event);
    }}
  }
  
  function startSequence() {
    cpp{{
    yaml_sequence_start_event_initialize(&self->event, NULL, NULL, 1,
        YAML_FLOW_SEQUENCE_STYLE);
    yaml_emitter_emit(&self->emitter, &self->event);
    }}
  }
  
  function endSequence() {
    cpp{{
    yaml_sequence_end_event_initialize(&self->event);
    yaml_emitter_emit(&self->emitter, &self->event);
    }}    
  }
  
  function scalar(x:Boolean) {
    auto value <- String(x);
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL,
        (yaml_char_t*)value.c_str(), value.length(), 1, 1,
        YAML_PLAIN_SCALAR_STYLE);
    yaml_emitter_emit(&self->emitter, &self->event);
    }}
  }

  function scalar(x:Integer) {
    auto value <- String(x);
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL,
        (yaml_char_t*)value.c_str(), value.length(), 1, 1,
        YAML_PLAIN_SCALAR_STYLE);
    yaml_emitter_emit(&self->emitter, &self->event);
    }}
  }

  function scalar(x:Real) {
    auto value <- String(x);
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL,
        (yaml_char_t*)value.c_str(), value.length(), 1, 1,
        YAML_PLAIN_SCALAR_STYLE);
    yaml_emitter_emit(&self->emitter, &self->event);
    }}
  }

  function scalar(value:String) {
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL,
        (yaml_char_t*)value.c_str(), value.length(), 1, 1,
        YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    yaml_emitter_emit(&self->emitter, &self->event);
    }}
  }
}
