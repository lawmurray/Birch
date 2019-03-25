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
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    yaml_document_start_event_initialize(&self->event, NULL, NULL, NULL, 1);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
    buffer.value.accept(this);
    cpp{{
    yaml_document_end_event_initialize(&self->event, 1);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    yaml_stream_end_event_initialize(&self->event);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    yaml_emitter_delete(&self->emitter);
    }}
    fclose(file);
  }

  function visit(value:ObjectValue) {
    cpp{{
    if (!yaml_mapping_start_event_initialize(&self->event, NULL, NULL, 1,
        YAML_FLOW_MAPPING_STYLE)) {
      error("generator error");
    }
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
    auto entry <- value.entries.walk();
    while entry? {
      auto e <- entry!;
      cpp{{
      yaml_scalar_event_initialize(&self->event, NULL, NULL,
          (yaml_char_t*)e->name.c_str(), e->name.length(), 1, 1,
          YAML_DOUBLE_QUOTED_SCALAR_STYLE);
      if (!yaml_emitter_emit(&self->emitter, &self->event)) {
        error("generator error");
      }
      }}
      e.buffer.value.accept(this);
    }
    cpp{{
    if (!yaml_mapping_end_event_initialize(&self->event)) {
      error("generator error");
    }
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
  }
  
  function visit(value:ArrayValue) {
    cpp{{
    yaml_sequence_start_event_initialize(&self->event, NULL, NULL, 1,
        YAML_FLOW_SEQUENCE_STYLE);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
    auto element <- value.buffers.walk();
    while element? {
      element!.value.accept(this);
    }
    cpp{{
    yaml_sequence_end_event_initialize(&self->event);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}    
  }

  function visit(value:StringValue) {
    auto v <- value.value;
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL, (yaml_char_t*)v.c_str(),
        v.length(), 1, 1, YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
  }

  function visit(value:RealValue) {
    auto v <- String(value.value);
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL, (yaml_char_t*)v.c_str(),
        v.length(), 1, 1, YAML_PLAIN_SCALAR_STYLE);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
  }

  function visit(value:IntegerValue) {
    auto v <- String(value.value);
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL, (yaml_char_t*)v.c_str(),
        v.length(), 1, 1, YAML_PLAIN_SCALAR_STYLE);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
  }

  function visit(value:BooleanValue) {
    auto v <- String(value.value);
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL, (yaml_char_t*)v.c_str(),
        v.length(), 1, 1, YAML_PLAIN_SCALAR_STYLE);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
  }

  function visit(value:NilValue) {
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL, (yaml_char_t*)"null",
        4, 1, 1, YAML_PLAIN_SCALAR_STYLE);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
  }
  
  function visit(value:BooleanVectorValue) {
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL, (yaml_char_t*)"null",
        4, 1, 1, YAML_PLAIN_SCALAR_STYLE);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
  }

  function visit(value:IntegerVectorValue) {
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL, (yaml_char_t*)"null",
        4, 1, 1, YAML_PLAIN_SCALAR_STYLE);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
  }
  
  function visit(value:RealVectorValue) {
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL, (yaml_char_t*)"null",
        4, 1, 1, YAML_PLAIN_SCALAR_STYLE);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
  }
  
  function visit(value:BooleanMatrixValue) {
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL, (yaml_char_t*)"null",
        4, 1, 1, YAML_PLAIN_SCALAR_STYLE);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
  }
  
  function visit(value:IntegerMatrixValue) {
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL, (yaml_char_t*)"null",
        4, 1, 1, YAML_PLAIN_SCALAR_STYLE);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
  }
  
  function visit(value:RealMatrixValue) {
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL, (yaml_char_t*)"null",
        4, 1, 1, YAML_PLAIN_SCALAR_STYLE);
    if (!yaml_emitter_emit(&self->emitter, &self->event)) {
      error("generator error");
    }
    }}
  }
}
