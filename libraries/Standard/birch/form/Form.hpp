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
