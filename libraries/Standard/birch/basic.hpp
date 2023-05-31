/**
 * @file
 */
#pragma once

namespace birch {
using Boolean = bool;
using Integer = int;
using Real = numbirch::real;
using String = std::string;
using File = FILE*;

template<class T> struct Random;

class Buffer_;
using Buffer = membirch::Shared<Buffer_>;

class Handler_;
using Handler = membirch::Shared<Handler_>;

class Delay_;
using Delay = membirch::Shared<Delay_>;

template<class T> class Distribution_;
template<class T> using Distribution = membirch::Shared<Distribution_<T>>;

template<class T> class Expression_;
template<class T> using Expression = membirch::Shared<Expression_<T>>;

template<class T> class BoxedValue_;
template<class T> using BoxedValue = membirch::Shared<BoxedValue_<T>>;

template<class T, class U> class BoxedForm_;
template<class T, class U> using BoxedForm = membirch::Shared<BoxedForm_<T,U>>;

class MoveVisitor_;
using MoveVisitor = membirch::Shared<MoveVisitor_>;

class ArgsVisitor_;
using ArgsVisitor = membirch::Shared<ArgsVisitor_>;

class RelinkVisitor_;
using RelinkVisitor = membirch::Shared<RelinkVisitor_>;

class GradVisitor_;
using GradVisitor = membirch::Shared<GradVisitor_>;

}
