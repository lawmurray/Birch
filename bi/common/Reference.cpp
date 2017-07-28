/**
 * @file
 */
#include "bi/common/Reference.hpp"

#include "bi/common/Overloaded.hpp"
#include "bi/expression/Identifier.hpp"
#include "bi/expression/Parameter.hpp"
#include "bi/expression/MemberParameter.hpp"
#include "bi/statement/GlobalVariable.hpp"
#include "bi/statement/LocalVariable.hpp"
#include "bi/statement/MemberVariable.hpp"
#include "bi/statement/Function.hpp"
#include "bi/statement/Coroutine.hpp"
#include "bi/statement/MemberFunction.hpp"
#include "bi/statement/BinaryOperator.hpp"
#include "bi/statement/UnaryOperator.hpp"
#include "bi/statement/AssignmentOperator.hpp"
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
template class bi::Reference<bi::Unknown>;
template class bi::Reference<bi::Parameter>;
template class bi::Reference<bi::MemberParameter>;
template class bi::Reference<bi::GlobalVariable>;
template class bi::Reference<bi::LocalVariable>;
template class bi::Reference<bi::MemberVariable>;
template class bi::Reference<bi::Function>;
template class bi::Reference<bi::Coroutine>;
template class bi::Reference<bi::MemberFunction>;
template class bi::Reference<bi::MemberCoroutine>;
template class bi::Reference<bi::BinaryOperator>;
template class bi::Reference<bi::UnaryOperator>;
template class bi::Reference<bi::AssignmentOperator>;
template class bi::Reference<bi::Basic>;
template class bi::Reference<bi::Class>;
template class bi::Reference<bi::Alias>;
template class bi::Reference<bi::Overloaded<bi::Function>>;
template class bi::Reference<bi::Overloaded<bi::Coroutine>>;
template class bi::Reference<bi::Overloaded<bi::MemberFunction>>;
template class bi::Reference<bi::Overloaded<bi::MemberCoroutine>>;
template class bi::Reference<bi::Overloaded<bi::BinaryOperator>>;
template class bi::Reference<bi::Overloaded<bi::UnaryOperator>>;
template class bi::Reference<bi::Overloaded<bi::AssignmentOperator>>;
