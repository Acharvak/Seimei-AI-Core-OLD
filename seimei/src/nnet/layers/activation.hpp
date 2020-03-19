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

#ifndef SEIMEI_NNET_LAYERS_ACTIVATION_HPP_
#define SEIMEI_NNET_LAYERS_ACTIVATION_HPP_

#include "common.hpp"
#include "nnet/framework.hpp"
#include "nnet/activations.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace seimei::nnet::layers {
/**
 * This layer will apply an activation function, but has no weights.
 * Header only.
 */
class ActivationBlueprint : public LayerBlueprint {
public:
	ActivationBlueprint(const string& activation) : name(""), activation(activation), outputSize(0) {
	}

	ActivationBlueprint() : ActivationBlueprint("") {
	}

	virtual ~ActivationBlueprint() = default;

	virtual unique_ptr<LayerBlueprint> clone() const override {
		return std::unique_ptr<ActivationBlueprint>(new ActivationBlueprint(name, activation, outputSize));
	}

	virtual unique_ptr<LayerBlueprint> makeShaped(
				const string& name,
				size_t input_size, size_t output_size) const override {
		if(activation.empty()) {
			throw std::logic_error("An Activation layer without an activation cannot have a shape");
		}
		if(input_size != output_size) {
			throw std::logic_error("An Activation layer must have the same input and output size");
		}
		assertNonZeroShape(name, output_size, output_size);
		return std::unique_ptr<LayerBlueprint>(new ActivationBlueprint(name, activation, output_size));
	}

	virtual bool getShape(string& name,
				size_t& input_size, size_t& output_size) const override {
		if(!outputSize) {
			return false;
		} else {
			name = this->name;
			input_size = outputSize;
			output_size = outputSize;
			return true;
		}
	}

	virtual void getMemoryRequirements(LayerMemoryRequirements& dest) const override {
		dest.numTempStateBackward = 0;
		dest.numTempStateForward = 0;
		dest.szDeltas = 0;
		dest.szInternalState = 0;
		dest.szPersistent = 0;
	}

	virtual void serialize(nlohmann::json& out) override {
		out["layer_type"] = "activation";
		out["activation"] = "activation";
	}

	virtual unique_ptr<LayerBlueprint> deserialize(const nlohmann::json& from) override {
		auto activation = from.at("activation").get<string>();
		return std::make_unique<ActivationBlueprint>(activation);
	}

	virtual shared_ptr<LayerPool> createPool(
				const string& network_name,
				H5::Group * weights_from) const override;	// Implementation below
protected:
	const string name, activation;
	const size_t outputSize;

	ActivationBlueprint(const string& name,
			const string& activation,
			size_t output_size) : name(name), activation(activation), outputSize(output_size) {
	}
};

class ActivationPool : public LayerPool {
	friend ActivationBlueprint;
	template<class Activation> friend class ActivationInstance;
public:
	virtual const string& getName() const override {
		return name;
	}

	virtual void initializeWeights([[maybe_unused]] array<uint64_t, 4>& state) override {
	}

	virtual void copyWeights([[maybe_unused]] const LayerPool& from) override {
		throw std::logic_error(".copyWeights called on an ActivationPool");
	}

	virtual void saveWeights([[maybe_unused]] H5::Group * to) const override {
	}

	virtual unique_ptr<LayerInstance> createInstance(
				[[maybe_unused]] bool trainable,
				FLOAT * input, FLOAT * output, [[maybe_unused]] FLOAT * temp) override {
		if(!outputSize) {
			throw std::logic_error("Cannot create an ActivationInstance without a shape");
		} else {
			return unique_ptr<LayerInstance>(createRawInstance(input, output));
		}
	}
protected:
	const size_t outputSize;
	const string name, activation;

	ActivationPool(const string& name, const string& activation, size_t output_size) : outputSize(output_size), name(name), activation(activation) {
	}

	LayerInstance* createRawInstance(FLOAT * input, FLOAT * output);
	// Implementation below
};

template<class Activation>
class ActivationInstance : public LayerInstance {
	friend ActivationPool;
public:
	ActivationInstance(ActivationInstance&) = delete;
	ActivationInstance(ActivationInstance&&) = delete;
	ActivationInstance& operator=(ActivationInstance&) = delete;
	ActivationInstance& operator=(ActivationInstance&&) = delete;
	virtual ~ActivationInstance() = default;

	virtual void forward() {
		std::transform(input, input + outputSize, output, [](FLOAT x) -> FLOAT {
			return Activation::call(x);
		});
	}

	virtual void backward([[maybe_unused]] FLOAT lrate) {
		std::transform(output, output + outputSize, input, input, [this](FLOAT nextder, FLOAT inp) -> FLOAT {
			auto x = nextder * Activation::derivative(inp);
			return antinan(x, antinanFactor);
		});
	}

	virtual void updateWeights([[maybe_unused]] FLOAT proportion) {
	}
protected:
	FLOAT *input, *output;
	size_t outputSize;
	FLOAT antinanFactor {DEFAULT_ANTINAN_FACTOR};

	ActivationInstance(FLOAT * input, FLOAT * output, size_t output_size) : input(input), output(output), outputSize(output_size) {
	}
};

inline shared_ptr<LayerPool> ActivationBlueprint::createPool([[maybe_unused]] const string& network_name, [[maybe_unused]] H5::Group * weights_from) const {
	return std::shared_ptr<LayerPool>(new ActivationPool(name, activation, outputSize));
}

inline LayerInstance* ActivationPool::createRawInstance(FLOAT * input, FLOAT * output) {
	if(activation == "identity") {
		return new ActivationInstance<activations::Identity>(input, output, outputSize);
	} else if(activation == "tanh") {
		return new ActivationInstance<activations::TanH>(input, output, outputSize);
	} else if(activation == "tanh/uz") {
		return new ActivationInstance<activations::Unzero<activations::TanH>>(input, output, outputSize);
	} else if(activation == "sigmoid") {
		return new ActivationInstance<activations::Sigmoid>(input, output, outputSize);
	} else {
		throw std::runtime_error(string("Activation not available for pure Activation layers (or at all): ") + activation);
	}
}
}

#endif
