hpp{{
#include <yaml.h>
}}

/**
 * Writer for JSON files.
 */
class JSONWriter < YAMLWriter {  
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
    /* the literals NaN, Infinity and -Infinity are not correct JSON, but are
     * fine for YAML, are correct JavaScript, and are supported by Python's
     * JSON module (also based on libyaml); so we encode to this */
    value:String;
    if x == inf {
      value <- "Infinity";
    } else if x == -inf {
      value <- "-Infinity";
    } else if isnan(x) {
      value <- "NaN";
    } else {
      value <- String(x);
    }
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

  function null() {
    cpp{{
    yaml_scalar_event_initialize(&self->event, NULL, NULL,
        (yaml_char_t*)"null", 4, 1, 1,
        YAML_PLAIN_SCALAR_STYLE);
    yaml_emitter_emit(&self->emitter, &self->event);
    }}
  }
}
