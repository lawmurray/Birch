/**
 * @file
 */
#pragma once

/*
 * Add `o.` to the start of each argument, e.g. `BIRCH_O_DOT(a, b, c)` yields
 * `o.a, o.b, o.c`.
 */
#define BIRCH_O_DOT3(arg) o.arg
#define BIRCH_O_DOT2(arg, ...) o.arg __VA_OPT__(, BIRCH_O_DOT3(__VA_ARGS__))
#define BIRCH_O_DOT1(arg, ...) o.arg __VA_OPT__(, BIRCH_O_DOT2(__VA_ARGS__))
#define BIRCH_O_DOT0(arg, ...) o.arg __VA_OPT__(, BIRCH_O_DOT1(__VA_ARGS__))
#define BIRCH_O_DOT(arg, ...) o.arg __VA_OPT__(, BIRCH_O_DOT0(__VA_ARGS__))

/*
 * Add `Integer ` to the start of each argument, e.g. `BIRCH_INT(a, b, c)` yields
 * `Integer a, Integer b, Integer c`.
 */
#define BIRCH_INT3(arg) Integer arg
#define BIRCH_INT2(arg, ...) Integer arg __VA_OPT__(, BIRCH_INT3(__VA_ARGS__))
#define BIRCH_INT1(arg, ...) Integer arg __VA_OPT__(, BIRCH_INT2(__VA_ARGS__))
#define BIRCH_INT0(arg, ...) Integer arg __VA_OPT__(, BIRCH_INT1(__VA_ARGS__))
#define BIRCH_INT(arg, ...) Integer arg __VA_OPT__(, BIRCH_INT0(__VA_ARGS__))

/*
 * Convert arguments to initializer list, e.g. `BIRCH_INIT(a, b, c)` yields
 * `a(a), b(b), c(c)`.
 */
#define BIRCH_INIT3(arg) arg(arg)
#define BIRCH_INIT2(arg, ...) arg(arg) __VA_OPT__(, BIRCH_INIT3(__VA_ARGS__))
#define BIRCH_INIT1(arg, ...) arg(arg) __VA_OPT__(, BIRCH_INIT2(__VA_ARGS__))
#define BIRCH_INIT0(arg, ...) arg(arg) __VA_OPT__(, BIRCH_INIT1(__VA_ARGS__))
#define BIRCH_INIT(arg, ...) arg(arg) __VA_OPT__(, BIRCH_INIT0(__VA_ARGS__))

/*
 * Convert arguments to copy initializer list, e.g. `BIRCH_COPY_INIT(a, b, c)`
 * yields `a(o.a), b(o.b), c(o.c)`.
 */
#define BIRCH_COPY_INIT3(arg) arg(o.arg)
#define BIRCH_COPY_INIT2(arg, ...) arg(o.arg) __VA_OPT__(, BIRCH_COPY_INIT3(__VA_ARGS__))
#define BIRCH_COPY_INIT1(arg, ...) arg(o.arg) __VA_OPT__(, BIRCH_COPY_INIT2(__VA_ARGS__))
#define BIRCH_COPY_INIT0(arg, ...) arg(o.arg) __VA_OPT__(, BIRCH_COPY_INIT1(__VA_ARGS__))
#define BIRCH_COPY_INIT(arg, ...) arg(o.arg) __VA_OPT__(, BIRCH_COPY_INIT0(__VA_ARGS__))

/*
 * Convert arguments to move initializer list, e.g. `BIRCH_COPY_INIT(a, b, c)`
 * yields `a(std::move(o.a)), b(std::move(o.b)), c(std::move(o.c))`.
 */
#define BIRCH_MOVE_INIT3(arg) arg(std::move(o.arg))
#define BIRCH_MOVE_INIT2(arg, ...) arg(std::move(o.arg)) __VA_OPT__(, BIRCH_MOVE_INIT3(__VA_ARGS__))
#define BIRCH_MOVE_INIT1(arg, ...) arg(std::move(o.arg)) __VA_OPT__(, BIRCH_MOVE_INIT2(__VA_ARGS__))
#define BIRCH_MOVE_INIT0(arg, ...) arg(std::move(o.arg)) __VA_OPT__(, BIRCH_MOVE_INIT1(__VA_ARGS__))
#define BIRCH_MOVE_INIT(arg, ...) arg(std::move(o.arg)) __VA_OPT__(, BIRCH_MOVE_INIT0(__VA_ARGS__))
