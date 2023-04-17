/**
 * @file
 */
#pragma once

namespace birch {
class Object_;

/**
 * Program function type.
 */
typedef int prog_t(int argc, char** argv);

/**
 * Default-constructible object factory function type.
 */
typedef Object_* fact_t();

/**
 * Retrieve a program function.
 *
 * @param name Program name.
 *
 * @return The program function, or `nullptr` if it does not exist.
 */
extern "C" prog_t* retrieve_program(const std::string& name);

/**
 * Register a program function.
 *
 * @param name Program name.
 * @param f Program function.
 *
 * @return Zero.
 */
int register_program(const std::string& name, prog_t* f);

/**
 * Retrieve a factory function.
 *
 * @param name Class name.
 *
 * @return The factory function, or `nullptr` if it does not exist.
 */
extern "C" fact_t* retrieve_factory(const std::string& name);

/**
 * Register a factory function.
 *
 * @param name Class name.
 * @param f Factory function.
 *
 * @return Zero.
 */
int register_factory(const std::string& name, fact_t* f);

}
