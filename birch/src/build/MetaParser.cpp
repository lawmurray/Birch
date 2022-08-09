/**
 * @file
 */
#include "src/build/MetaParser.hpp"

#include "src/exception/DriverException.hpp"
#include "src/primitive/system.hpp"

birch::MetaParser::MetaParser() {
  fs::path path;
  if (fs::exists("birch.yaml")) {
    path = "birch.yaml";
  } else if (fs::exists("birch.yml")) {
    path = "birch.yml";
  } else if (fs::exists("birch.json")) {
    path = "birch.json";
  } else if (fs::exists("META.yaml")) {
    warn("the preferred name for the build configuration file is now birch.yaml");
    path = "META.yaml";
  } else if (fs::exists("META.yml")) {
    warn("the preferred name for the build configuration file is now birch.yml");
    path = "META.yml";
  } else if (fs::exists("META.json")) {
    warn("the preferred name for the build configuration file is now birch.json");
    path = "META.json";
  } else {
    throw DriverException(std::string("no build configuration file; create a file named ") +
        "birch.yaml, birch.yml or birch.json.");
  }
  file = fopen(path.string().c_str(), "r");
  if (!file) {
    throw DriverException("could not open file " + path.string());
  }  
}

birch::MetaParser::~MetaParser() {
  fclose(file);
}

birch::MetaParser::map_type birch::MetaParser::parse() {
  yaml_parser_initialize(&parser);
  yaml_parser_set_input_file(&parser, file);
  int done = 0;
  while (!done) {
    if (!yaml_parser_parse(&parser, &event)) {
      throw DriverException("syntax error in build configuration file.");
    }
    if (event.type == YAML_SEQUENCE_START_EVENT) {
      parseSequence();
    } else if (event.type == YAML_MAPPING_START_EVENT) {
      parseMapping();
    } else {
      done = event.type == YAML_STREAM_END_EVENT;
      yaml_event_delete(&event);
    }
  }
  yaml_parser_delete(&parser);
  return contents;
}

void birch::MetaParser::parseMapping() {
  yaml_event_delete(&event);
  int done = 0;
  while (!done) {
    /* read one name/value pair on each iteration */
    if (!yaml_parser_parse(&parser, &event)) {
      throw DriverException("syntax error in build configuration file.");
    }
    if (event.type == YAML_SCALAR_EVENT) {
      /* key */
      auto data = (char*)event.data.scalar.value;
      auto length = event.data.scalar.length;
      std::string key(data, length);
      yaml_event_delete(&event);

      if (keys.empty()) {
        keys.push(key);
      } else {
        keys.push(keys.top() + "." + key);
      }
      
      /* value */
      if (!yaml_parser_parse(&parser, &event)) {
        throw DriverException("syntax error in build configuration file.");
      }
      if (event.type == YAML_SCALAR_EVENT) {
        parseScalar();
      } else if (event.type == YAML_SEQUENCE_START_EVENT) {
        parseSequence();
      } else if (event.type == YAML_MAPPING_START_EVENT) {
        parseMapping();
      } else {
        yaml_event_delete(&event);
      }

      keys.pop();
    } else {
      done = event.type == YAML_MAPPING_END_EVENT;
      yaml_event_delete(&event);
    }
  }
}

void birch::MetaParser::parseSequence() {
  yaml_event_delete(&event);
  int done = 0;
  while (!done) {
    if (!yaml_parser_parse(&parser, &event)) {
      throw DriverException("syntax error in build configuration file.");
    }
    if (event.type == YAML_SCALAR_EVENT) {
      parseScalar();
    } else if (event.type == YAML_SEQUENCE_START_EVENT) {
      parseSequence();
    } else if (event.type == YAML_MAPPING_START_EVENT) {
      parseMapping();
    } else {
      done = event.type == YAML_SEQUENCE_END_EVENT;
      yaml_event_delete(&event);
    }
  }
}

void birch::MetaParser::parseScalar() {
  auto data = (char*)event.data.scalar.value;
  auto length = event.data.scalar.length;
  std::string value(data, length);

  contents[keys.top()].push_back(value);
  yaml_event_delete(&event);
}
