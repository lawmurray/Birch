/**
 * @file
 */
#pragma once

#include "bi/common/Located.hpp"

namespace bi {
class Visitor;

/**
 * Name.
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

  /**
   * Counter for unique names.
   */
  static int COUNTER;
};
}
