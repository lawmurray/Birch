cpp{{
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
    cpp{{
    /* slurp in the whole file */
    std::ifstream stream(path);
    std::string input;
    getline(stream, input,
        std::string::traits_type::to_char_type(
        std::string::traits_type::eof()));    
    stream.close();
    
    libbirch::Optional<String> key;
    libbirch::Shared<Value> value;
    
    yaml_parser_t parser;
    yaml_event_t event;
    yaml_parser_initialize(&parser);
    yaml_parser_set_input_string(&parser,
        (const unsigned char*)input.c_str(), input.length());
    int done = 0;
    while (!done) {
      if (yaml_parser_parse(&parser, &event)) {
        if (event.type == YAML_SCALAR_EVENT) {
          char* data = (char*)event.data.scalar.value;
          //char* tag = (char*)event.data.scalar.tag;
          size_t length = event.data.scalar.length;
          char* endptr;
          
          if (stack->top()->isObject() && !key.query()) {
            /* is a key */
            key = std::string(data, length);
          } else {
            /* is a value */
            auto intValue = std::strtol(data, &endptr, 10);
            if (endptr == data + length) {
              value = bi::IntegerValue(intValue);
            } else {
              auto doubleValue = std::strtod(data, &endptr);
              if (endptr == data + length) {
                value = bi::RealValue(doubleValue);
              } else if (std::strcmp(data, "true") == 0) {
                value = bi::BooleanValue(true);
              } else if (std::strcmp(data, "false") == 0) {
                value = bi::BooleanValue(false);
              } else if (std::strcmp(data, "null") == 0) {
                value = bi::NilValue();
              } else {
                value = bi::StringValue(std::string(data, length));
              }
            }
            if (stack->empty()) {
              stack->push(value);
            } else if (stack->top()->isObject()) {
              stack->top()->push(key.get(), value);
              key = libbirch::nil;
            } else {
              stack->top()->push(value);
            }
          }
        } else if (event.type == YAML_SEQUENCE_START_EVENT) {
          stack->push(bi::ArrayValue());
        } else if (event.type == YAML_SEQUENCE_END_EVENT) {
          stack->pop();
        } else if (event.type == YAML_MAPPING_START_EVENT) {
          stack->push(bi::ObjectValue());
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
    if stack.empty() {
      return NilValue();
    } else {
      assert stack.size() == 1;
      return stack.top();
    }
  }
}
