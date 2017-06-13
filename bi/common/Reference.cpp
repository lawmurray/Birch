/**
 * @file
 */
#include "bi/common/Reference.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/statement/Function.hpp"
#include "bi/statement/BinaryOperator.hpp"
#include "bi/statement/UnaryOperator.hpp"
#include "bi/statement/AssignmentOperator.hpp"
#include "bi/type/TypeParameter.hpp"
#include "bi/statement/Program.hpp"

template<class Target>
bi::Reference<Target>::Reference(const Target* target) :
    target(target) {
  //
}

template<class Target>
bi::Reference<Target>::~Reference() {
  //
}

/*
 * Explicit instantiations.
 */
template class bi::Reference<bi::Parameter>;
template class bi::Reference<bi::Function>;
template class bi::Reference<bi::BinaryOperator>;
template class bi::Reference<bi::UnaryOperator>;
template class bi::Reference<bi::AssignmentOperator>;
template class bi::Reference<bi::Program>;
template class bi::Reference<bi::TypeParameter>;
