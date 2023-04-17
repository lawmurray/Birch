/**
 * @file
 */
#include "birch/registry.hpp"

#include <map>

namespace birch {
/**
 * Program function registry.
 */
static std::map<std::string,prog_t*> programs;

/**
 * Default-constructible object factory function registry.
 */
static std::map<std::string,fact_t*> factories;

extern "C" prog_t* retrieve_program(const std::string& name) {
  auto iter = programs.find(name);
  if (iter != programs.end()) {
    return iter->second;
  } else {
    return nullptr;
  }
}

int register_program(const std::string& name, prog_t* f) {
  programs[name] = f;
  return 0;
}

extern "C" fact_t* retrieve_factory(const std::string& name) {
  auto iter = factories.find(name);
  if (iter != factories.end()) {
    return iter->second;
  } else {
    return nullptr;
  }
}

int register_factory(const std::string& name, fact_t* f) {
  factories[name] = f;
  return 0;
}

}
