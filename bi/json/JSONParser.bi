hpp{{
#include <yaml.h>
}}

/**
 * Parser for JSON files.
 */
class JSONParser < Parser {
  function parse(path:String) -> Value? {
    stack:Stack<Value>;
    input:String;
    cpp{{
    yaml_parser_t parser;
    yaml_event_t event;
    yaml_parser_initialize(&parser);
    yaml_parser_set_input_string(&parser, (const unsigned char*)input.c_str(), input.length());
    int done = 0;
    while (!done) {
      if (!yaml_parser_parse(&parser, &event)) {
        switch (event.type) {
          case YAML_NO_EVENT:
          case YAML_STREAM_START_EVENT:
          case YAML_STREAM_END_EVENT:
          case YAML_DOCUMENT_START_EVENT:
          case YAML_DOCUMENT_END_EVENT:
          case YAML_ALIAS_EVENT:
            break;
          case YAML_SCALAR_EVENT:
            break;
          case YAML_SEQUENCE_START_EVENT:
            stack->push(ArrayValue::create_());
            break;
          case YAML_SEQUENCE_END_EVENT:
            stack->pop();
            break;
          case YAML_MAPPING_START_EVENT:
            stack->push(ObjectValue::create_());
            break;
          case YAML_MAPPING_END_EVENT:
            stack->pop();
            break;
        }
      }
      done = (event.type == YAML_STREAM_END_EVENT);
      yaml_event_delete(&event);
    }
    yaml_parser_delete(&parser);
    }}
    return stack.top();
  }
}
