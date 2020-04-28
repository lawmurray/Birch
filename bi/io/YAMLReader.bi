hpp{{
#include <yaml.h>
}}

/**
 * Reader for YAML files.
 */
class YAMLReader < Reader {
  /**
   * The file.
   */
  file:File;

  hpp{{
  yaml_parser_t parser;
  yaml_event_t event;
  }}

  function open(path:String) {
    file <- fopen(path, READ);
  }

  function scan() -> MemoryBuffer {
    buffer:MemoryBuffer;
    cpp{{
    yaml_parser_initialize(&self_()->parser);
    yaml_parser_set_input_file(&self_()->parser, self_()->file);
    int done = 0;
    while (!done) {
      if (!yaml_parser_parse(&self_()->parser, &self_()->event)) {
        error("parse error");
      }
      if (self_()->event.type == YAML_SEQUENCE_START_EVENT) {
        self_()->parseSequence(buffer);
      } else if (self_()->event.type == YAML_MAPPING_START_EVENT) {
        self_()->parseMapping(buffer);
      } else {
        done = self_()->event.type == YAML_STREAM_END_EVENT;
        yaml_event_delete(&self_()->event);
      }
    }
    yaml_parser_delete(&self_()->parser);
    }}
    return buffer;
  }

  fiber walk() -> Buffer {
    auto done <- false;
    cpp{{
    yaml_parser_initialize(&self_()->parser);
    yaml_parser_set_input_file(&self_()->parser, self_()->file);
    while (!done) {
      if (!yaml_parser_parse(&self_()->parser, &self_()->event)) {
        error("parse error");
      } else if (self_()->event.type == YAML_MAPPING_START_EVENT) {
        error("not a sequential file");
      } else if (self_()->event.type == YAML_SEQUENCE_START_EVENT ||
          self_()->event.type == YAML_STREAM_END_EVENT) {
        done = true;
      } else {
        yaml_event_delete(&self_()->event);
      }
    }
    done = self_()->event.type == YAML_STREAM_END_EVENT;
    yaml_event_delete(&self_()->event);
    }}
    while !done {
      buffer:MemoryBuffer;
      substantial:Boolean <- false;
      cpp{{
      if (!yaml_parser_parse(&self_()->parser, &self_()->event)) {
        error("parse error");
      }
      if (self_()->event.type == YAML_SCALAR_EVENT) {
        self_()->parseScalar(buffer);
        substantial = true;
      } else if (self_()->event.type == YAML_SEQUENCE_START_EVENT) {
        self_()->parseSequence(buffer);
        substantial = true;
      } else if (self_()->event.type == YAML_MAPPING_START_EVENT) {
        self_()->parseMapping(buffer);
        substantial = true;
      } else {
        done = self_()->event.type == YAML_SEQUENCE_END_EVENT;
        yaml_event_delete(&self_()->event);
      }
      }}
      if substantial {
        yield buffer;
      }
    }
    cpp{{
    yaml_parser_delete(&self_()->parser);
    }}
  }

  function close() {
    fclose(file);
  }
  
  function parseMapping(buffer:Buffer) {
    buffer.setObject();
    cpp{{
    yaml_event_delete(&self_()->event);
    int done = 0;
    while (!done) {
      /* read one name/value pair on each iteration */
      if (!yaml_parser_parse(&self_()->parser, &self_()->event)) {
        error("parse error");
      }
      if (self_()->event.type == YAML_SCALAR_EVENT) {
        /* name */
        char* data = (char*)self_()->event.data.scalar.value;
        size_t length = self_()->event.data.scalar.length;
        std::string name(data, length);
        yaml_event_delete(&self_()->event);
        
        /* value */
        if (!yaml_parser_parse(&self_()->parser, &self_()->event)) {
          error("parse error");
        }
        if (self_()->event.type == YAML_SCALAR_EVENT) {
          self_()->parseScalar(buffer->setChild(name));
        } else if (self_()->event.type == YAML_SEQUENCE_START_EVENT) {
          self_()->parseSequence(buffer->setChild(name));
        } else if (self_()->event.type == YAML_MAPPING_START_EVENT) {
          self_()->parseMapping(buffer->setChild(name));
        } else {
          buffer->setChild(name);
          yaml_event_delete(&self_()->event);
        }
      } else {
        done = self_()->event.type == YAML_MAPPING_END_EVENT;
        yaml_event_delete(&self_()->event);
      }
    }
    }}
  }
  
  function parseSequence(buffer:Buffer) {
    buffer.setArray();
    cpp{{
    yaml_event_delete(&self_()->event);
    int done = 0;
    while (!done) {
      if (!yaml_parser_parse(&self_()->parser, &self_()->event)) {
        error("parse error");
      }
      if (self_()->event.type == YAML_SCALAR_EVENT) {
        self_()->parseScalar(buffer->push());
      } else if (self_()->event.type == YAML_SEQUENCE_START_EVENT) {
        self_()->parseSequence(buffer->push());
      } else if (self_()->event.type == YAML_MAPPING_START_EVENT) {
        self_()->parseMapping(buffer->push());
      } else {
        done = self_()->event.type == YAML_SEQUENCE_END_EVENT;
        yaml_event_delete(&self_()->event);
      }
    }
    }}
  }
  
  function parseScalar(buffer:Buffer) {
    cpp{{
    auto data = (char*)self_()->event.data.scalar.value;
    auto length = self_()->event.data.scalar.length;
    auto endptr = data;
    
    auto intValue = std::strtoll(data, &endptr, 10);
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
      } else if (std::strcmp(data, "Infinity") == 0) {
        buffer->setReal(std::numeric_limits<Real>::infinity());
      } else if (std::strcmp(data, "-Infinity") == 0) {
        buffer->setReal(-std::numeric_limits<Real>::infinity());
      } else if (std::strcmp(data, "NaN") == 0) {
        buffer->setReal(std::numeric_limits<Real>::quiet_NaN());
      } else {
        buffer->setString(std::string(data, length));
      }
    }
    yaml_event_delete(&self_()->event);
    }}
  }
}
