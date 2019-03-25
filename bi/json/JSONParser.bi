hpp{{
#include <yaml.h>
}}

/**
 * Parser for JSON files.
 */
class JSONParser < Parser {
  hpp{{
  yaml_parser_t parser;
  yaml_event_t event;
  }}

  function parse(path:String, buffer:Buffer) {
    auto file <- fopen(path, READ);
    cpp{{
    yaml_parser_initialize(&self->parser);
    yaml_parser_set_input_file(&self->parser, file);
    int done = 0;
    while (!done) {
      if (!yaml_parser_parse(&self->parser, &self->event)) {
        error("parse error");
      }
      if (event.type == YAML_SEQUENCE_START_EVENT) {
        self->parseSequence(buffer);
      } else if (event.type == YAML_MAPPING_START_EVENT) {
        self->parseMapping(buffer);
      } else {
        done = event.type == YAML_STREAM_END_EVENT;
        yaml_event_delete(&self->event);
      }
    }
    yaml_parser_delete(&self->parser);
    }}
    fclose(file);
  }
  
  function parseMapping(buffer:Buffer) {
    buffer.setObject();
    cpp{{
    yaml_event_delete(&self->event);
    int done = 0;
    while (!done) {
      /* read one name/value pair on each iteration */
      if (!yaml_parser_parse(&self->parser, &self->event)) {
        error("parse error");
      }
      if (event.type == YAML_SCALAR_EVENT) {
        /* name */
        char* data = (char*)event.data.scalar.value;
        size_t length = event.data.scalar.length;
        std::string name(data, length);
        yaml_event_delete(&self->event);
        
        /* value */
        if (!yaml_parser_parse(&self->parser, &self->event)) {
          error("parse error");
        }
        if (event.type == YAML_SCALAR_EVENT) {
          self->parseScalar(buffer->setChild(name));
        } else if (event.type == YAML_SEQUENCE_START_EVENT) {
          self->parseSequence(buffer->setChild(name));
        } else if (event.type == YAML_MAPPING_START_EVENT) {
          self->parseMapping(buffer->setChild(name));
        } else {
          buffer->setChild(name);
          yaml_event_delete(&self->event);
        }
      } else {
        done = event.type == YAML_MAPPING_END_EVENT;
        yaml_event_delete(&self->event);
      }
    }
    }}
  }
  
  function parseSequence(buffer:Buffer) {
    buffer.setArray();
    cpp{{
    yaml_event_delete(&self->event);
    int done = 0;
    while (!done) {
      if (!yaml_parser_parse(&self->parser, &self->event)) {
        error("parse error");
      }
      if (event.type == YAML_SCALAR_EVENT) {
        self->parseScalar(buffer->push());
      } else if (event.type == YAML_SEQUENCE_START_EVENT) {
        self->parseSequence(buffer->push());
      } else if (event.type == YAML_MAPPING_START_EVENT) {
        self->parseMapping(buffer->push());
      } else {
        done = event.type == YAML_SEQUENCE_END_EVENT;
        yaml_event_delete(&self->event);
      }
    }
    }}
  }
  
  function parseScalar(buffer:Buffer) {
    cpp{{
    char* data = (char*)event.data.scalar.value;
    size_t length = event.data.scalar.length;
    char* endptr;
    
    auto intValue = std::strtol(data, &endptr, 10);
    if (endptr == data + length) {
      buffer->setInteger(intValue);
    } else {
      auto realValue = std::strtod(data, &endptr);
      if (endptr == data + length) {
        buffer->setReal(realValue);
      } else if (std::strcmp(data, "true") == 0) {
        buffer->setBoolean(true);
      } else if (std::strcmp(data, "false") == 0) {
        buffer->setBoolean(false);
      } else if (std::strcmp(data, "null") == 0) {
        buffer->setNil();
      } else {
        buffer->setString(std::string(data, length));
      }
    }
    yaml_event_delete(&self->event);
    }}
  }
}
