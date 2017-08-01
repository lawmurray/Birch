/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

#include <string>
#include <cstring>

namespace bi {
class Visitor;

/**
 * Name.
 */
class Name {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   */
  Name(const std::string& name);

  /**
   * Constructor.
   *
   * @param name Name.
   */
  Name(const char* name = "");

  /**
   * Constructor.
   *
   * @param name Name.
   */
  Name(const char name);

  /**
   * Destructor.
   */
  virtual ~Name();

  /**
   * Get name as string.
   */
  const std::string& str() const;

  /**
   * Is this name non-empty?
   */
  bool isEmpty() const;

  virtual void accept(Visitor* visitor) const;

  virtual bool operator==(const Name& o) const;
  bool operator!=(const Name& o) const;

private:
  /**
   * Name.
   */
  std::string name;
};
}
