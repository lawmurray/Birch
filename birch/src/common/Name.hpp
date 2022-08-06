/**
 * @file
 */
#pragma once

#include "src/common/Located.hpp"

namespace birch {
class Visitor;

/**
 * Name.
 *
 * @ingroup common
 *
 * Name objects may be shared between multiple objects in the abstract
 * syntax tree, they need not be uniquely assigned.
 */
class Name {
public:
  /**
   * Default constructor. Generates a package-wide unique name.
   */
  Name();

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
  Name(const char* name);

  /**
   * Constructor.
   *
   * @param name Name.
   */
  Name(const char name);

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

  /**
   * Counter for unique names.
   */
  static int COUNTER;
};
}
