/**
 * @file
 */
#pragma once

#include "src/birch.hpp"

namespace birch {
/**
 * Parser for YAML files.
 */
class MetaParser {
public:
  using map_type = std::map<std::string,std::list<std::string>>;

  /**
   * Constructor.
   */
  MetaParser();

  /**
   * Destructor.
   */
  ~MetaParser();

  /**
   * Parse the file.
   * 
   * @return The contents of the file.
   */
  map_type parse();

private:
  void parseMapping();
  void parseSequence();
  void parseScalar();

  /**
   * LibYAML parser.
   */
  yaml_parser_t parser;

  /**
   * LibYAML event.
   */
  yaml_event_t event;

  /**
   * File.
   */
  FILE* file;

  /**
   * Stack of keys,
   */
  std::stack<std::string> keys;

  /**
   * Contents, populated during parse.
   */
  map_type contents;
};
}
