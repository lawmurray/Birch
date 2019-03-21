hpp{{
#include <yaml.h>

#include <iostream>
#include <fstream>
}}

/**
 * Parser for JSON files.
 */
class JSONParser < Parser {
  function parse(path:String) -> Value? {
    stack:Stack<Value>;
    value:NilValue;
    cpp{{
    /* slurp in the whole file */
    std::ifstream stream(path);
    std::string input;
    getline(stream, input,
        std::string::traits_type::to_char_type(
        std::string::traits_type::eof()));    
    stream.close();
    
    yaml_parser_t parser;
    yaml_event_t event;
    yaml_parser_initialize(&parser);
    yaml_parser_set_input_string(&parser, (const unsigned char*)input.c_str(), input.length());
    int done = 0;
    while (!done) {
      if (yaml_parser_parse(&parser, &event)) {
        if (event.type == YAML_SCALAR_EVENT) {
		  const char* type = "string";
		  char* data = (char*)event.data.scalar.value;
          char* tag = (char*)event.data.scalar.tag;
		  size_t length = event.data.scalar.length;
		  char* endptr;
		  auto intValue = std::strtol(data, &endptr, 10);
		  if (endptr == data + length) {
		    type = "int";
		  } else {
		    auto doubleValue = std::strtod(data, &endptr);
		    if (endptr == data + length) {
		      type = "double";
		    } else if (strcmp(data, "true") == 0) {
		      type = "bool";
		    } else if (strcmp(data, "false") == 0) {
		      type = "bool";
		    } else if (strcmp(data, "null") == 0) {
		      type = "null";
		    }
		  }
		  printf("value: %s, tag: %s, type: %s\n", data, tag, type);
        } else if (event.type == YAML_SEQUENCE_START_EVENT) {
          stack->push(ArrayValue::create_());
        } else if (event.type == YAML_SEQUENCE_END_EVENT) {
          stack->pop();
        } else if (event.type == YAML_MAPPING_START_EVENT) {
          stack->push(ObjectValue::create_());
        } else if (event.type == YAML_MAPPING_END_EVENT) {
          stack->pop();
        }
      } else {
        error("syntax error");
      }      
      done = (event.type == YAML_STREAM_END_EVENT);
      yaml_event_delete(&event);
    }
    yaml_parser_delete(&parser);
    }}
    return value;
  }
}
