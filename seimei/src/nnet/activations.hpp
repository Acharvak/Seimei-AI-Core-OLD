/**********************************************************************
This file is part of the Seimei AI Project:
	https://github.com/Acharvak/Seimei-AI

Copyright 2020 Fedor Uvarov

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
**********************************************************************/
// SPDX-License-Identifier: Apache-2.0

/**
 * This file defines a number of activation functions for use in
 * neural networks. They are all inline due to simplicity.
 */

#ifndef SEIMEI_NNET_ACTIVATIONS_HPP_
#define SEIMEI_NNET_ACTIVATIONS_HPP_

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace seimei::nnet::activations {
class Identity {
public:
	static FLOAT call(FLOAT x) {
		return x;
	}
	static FLOAT derivative([[maybe_unused]] FLOAT x) {
		return F1;
	}
	static const std::string getName() {
		return name;
	}
private:
	inline static const std::string name {"f(x) = x"};
};

class TanH {
public:
	static FLOAT call(FLOAT x) {
		return std::tanh(x);
	}
	static FLOAT derivative(FLOAT x) {
		x = std::cosh(x);
		return F1 / (x * x);
	}
	static const std::string getName() {
		return name;
	}
private:
	inline static const std::string name {"tanh"};
};

class Sigmoid {
public:
	static FLOAT call(FLOAT x) {
		return F1 / (F1 + std::exp(-x));
	}
	static FLOAT derivative(FLOAT x) {
		auto a = std::exp(x);
		auto b = (F1 + a);
		return a / (b * b);
	}
	static const std::string getName() {
		return name;
	}
private:
	inline static const std::string name {"f(x) = 1 / (1 + e ** -x)"};
};

/// This is a wrapper for another activation that changes zeros to epsilons
template<class Activation> class Unzero {
public:
	static FLOAT call(FLOAT x) {
		x = Activation::call(x);
		if(!x) {
			return std::numeric_limits<FLOAT>::epsilon();
		} else {
			return x;
		}
	}
	static FLOAT derivative(FLOAT x) {
		return Activation::derivative(x);
	}
	static const std::string getName() {
		return Activation::getName() + " -> unzero";
	}
};

/**
 * This is a wrapper for another activation that clamps the result
 * between the minimum and maximum representable finite values.
 *
 * Arbitrary limits are not supported because the function would
 * be difficult to derive otherwise.
 */
template<class Activation> class ClampExtremes {
public:
	static FLOAT call(FLOAT x) {
		return std::clamp(Activation::call(x),
				std::numeric_limits<FLOAT>::lowest(),
				std::numeric_limits<FLOAT>::max());
	}

	static FLOAT derivative(FLOAT x) {
		// This should be a sufficient approximation
		return std::clamp(Activation::derivative(x),
						std::numeric_limits<FLOAT>::lowest(),
						std::numeric_limits<FLOAT>::max());
	}

	static const std::string getName() {
		return Activation::getName() + " -> make finite";
	}
};
}

#endif
