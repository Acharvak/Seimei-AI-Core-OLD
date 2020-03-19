/*  Written in 2019 by David Blackman and Sebastiano Vigna

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>.

Adapted from the original code by Fedor Uvarov, 2019
Original code at:
    http://prng.di.unimi.it/xoshiro256plusplus.c

The same permissions as declared above apply to the modified code. */
// SPDX-License-Identifier: CC0-1.0

/**
 * This is the xoshiro256++ 1.0 random number generator. Header-only due
 * to simplicity.
 */

#ifndef SEIMEI_XOSHIROPP_HPP_
#define SEIMEI_XOSHIROPP_HPP_

#include "common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace seimei {
namespace {
	namespace {
		inline uint64_t _xoshiropp_rotl(const uint64_t x, int k) {
			return (x << k) | (x >> (64 - k));
		}
	}

/// Return the next uint64_t and change the generator state
inline uint64_t xoshiropp(std::array<uint64_t, 4>& state) {
	const uint64_t result = _xoshiropp_rotl(state[0] + state[3], 23) + state[0];

	const uint64_t t = state[1] << 17;

	state[2] ^= state[0];
	state[3] ^= state[1];
	state[1] ^= state[2];
	state[0] ^= state[3];

	state[2] ^= t;

	state[3] = _xoshiropp_rotl(state[3], 45);

	return result;
}

/// Convert a random int to a signed floating-point value
/// TODO: ensure that it's precise? That's why it's a separate function.
inline FLOAT from_int(uint64_t value) {
	// 52 bits
	double x = static_cast<double>(value & 0xFFFFFFFFFFFFF) / static_cast<double>(0xFFFFFFFFFFFFF);
	return static_cast<FLOAT>(x);
}
}
}

#endif
