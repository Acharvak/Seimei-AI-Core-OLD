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

// DenseInstance implementation - template class

#ifndef SEIMEI_NNET_LAYERS_DENSE_HPP_
#include "nnet/layers/dense.hpp"
#endif

#include "xoshiropp.hpp"
#include "blas.hpp"
#include "nnet/activations.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace seimei::nnet::layers {
inline DenseBlueprint::DenseBlueprint(const string& name,
		const string& activation,
		size_t input_size, size_t output_size) :
				name(name), activation(activation),
				inputSize(input_size), outputSize(output_size) {
	// make_shaped will check if the activation is valid
	// (for API simplicity)
}

inline DenseBlueprint::DenseBlueprint() : DenseBlueprint("", "", 0, 0) {
}

inline DenseBlueprint::DenseBlueprint(const string& activation) :
		DenseBlueprint("", activation, 0, 0) {
}

inline unique_ptr<LayerBlueprint> DenseBlueprint::clone() const {
	return unique_ptr<LayerBlueprint>(new DenseBlueprint(*this));
}

inline unique_ptr<LayerBlueprint> DenseBlueprint::makeShaped(
		const string& name,
		size_t input_size, size_t output_size) const {
	if(activation.empty()) {
		throw std::logic_error("A DenseBlueprint without activation cannot have a shape");
	}
	assertNonZeroShape(name, input_size, output_size);
	assertSZMUL(input_size, output_size);
	return unique_ptr<LayerBlueprint>(
			new DenseBlueprint(name, this->activation,
					input_size, output_size));
}

inline bool DenseBlueprint::getShape(string& name,
		size_t& input_size, size_t& output_size) const {
	if(!inputSize) {
		return false;
	} else {
		name = this->name;
		input_size = inputSize;
		output_size = outputSize;
		return true;
	}
}

inline void DenseBlueprint::getMemoryRequirements(
		LayerMemoryRequirements& dest) const {
	assertNonZeroShape(name, inputSize, outputSize);
	dest.numTempStateForward = 0;
	dest.numTempStateBackward = 0;
	dest.szPersistent = inputSize * outputSize * sizeof(FLOAT);
	// On backward runs we'll have an internal buffer for previous output
	dest.szDeltas = dest.szPersistent + outputSize * sizeof(FLOAT);
	dest.szInternalState = 0;
}

inline void DenseBlueprint::serialize(nlohmann::json& out) {
	out["layer_type"] = "dense";
	out["activation"] = activation;
}

inline unique_ptr<LayerBlueprint> DenseBlueprint::deserialize(const nlohmann::json& from) {
	auto activation = from.at("activation").get<string>();
	return std::make_unique<DenseBlueprint>(activation);
}

inline shared_ptr<LayerPool> DenseBlueprint::createPool(
				const string& network_name,
				H5::Group * weights_from) const {
	auto result = new DensePool(name, network_name, activation,
			inputSize, outputSize);
	result->loadWeights(weights_from);
	return shared_ptr<LayerPool>(result);
}

inline DensePool::DensePool(const string& name,
		const string& network_name,
		const string& activation,
		size_t input_size, size_t output_size) :
				inputSize(input_size), outputSize(output_size),
				name(name),
				networkName(network_name), activation(activation),
				weights(vector<FLOAT>(output_size * inputSize)) {
}

inline const string& DensePool::getName() const {
	return name;
}

inline void DensePool::loadWeights(H5::Group * weights_from) {
	if(weights_from) {
		throw std::logic_error("Loading weights from file not implemented yet");
	}
}

inline void DensePool::initializeWeights(array<uint64_t, 4>& state) {
	for(FLOAT& w : weights) {
		w = from_int(xoshiropp(state)) - 0.5f;
	}
}

inline void DensePool::copyWeights([[maybe_unused]] const LayerPool& from) {
	throw std::logic_error("Copying weights is not yet implemented for Dense layers");
}

inline void DensePool::saveWeights([[maybe_unused]] H5::Group * to) const {
	throw std::logic_error("Saving weights is not yet implemented for Dense layers");
}

template<class Activation> LayerInstance * DensePool::createRawInstance(
			bool trainable, FLOAT * input, FLOAT * output) {
	if(trainable) {
		return new TrainableDenseInstance<Activation>(shared_from_this(), input, output,
				inputSize, outputSize);
	} else {
		return new DenseInstance<Activation>(shared_from_this(), input, output,
				inputSize, outputSize);
	}
}

inline unique_ptr<LayerInstance> DensePool::createInstance(
				bool trainable,
				FLOAT * input, FLOAT * output, [[maybe_unused]] FLOAT * temp) {
	LayerInstance * result;
	if(activation == "identity") {
		result = createRawInstance<activations::Identity>(trainable, input, output);
	} else if(activation == "tanh") {
		result = createRawInstance<activations::TanH>(trainable, input, output);
	} else if(activation == "tanh/uz") {
		result = createRawInstance<activations::Unzero<activations::TanH>>(trainable, input, output);
	} else if(activation == "sigmoid") {
		result = createRawInstance<activations::Sigmoid>(trainable, input, output);
	} else {
		throw std::runtime_error(string("Activation not available for Dense layers (or at all): ")
				+ activation);
	}
	return unique_ptr<LayerInstance>(result);
}

template<class Activation> DenseInstance<Activation>::DenseInstance(
		shared_ptr<DensePool> pool,
		FLOAT * input, FLOAT * output,
		size_t input_size, size_t output_size) :
				pool(pool), input(input), output(output),
				inputSize(input_size), outputSize(output_size) {
}

template<class Activation> TrainableDenseInstance<Activation>::TrainableDenseInstance(
		shared_ptr<DensePool> pool,
		FLOAT * input, FLOAT * output,
		size_t input_size, size_t output_size) :
			DenseInstance<Activation>(pool, input, output, input_size, output_size),
			deltas(input_size * output_size), outBuffer(output_size) {
}

template<class Activation> void DenseInstance<Activation>::forward() {
	gemv(pool->weights.data(), outputSize, inputSize, false, 1, input, 0, output);
	std::transform(output, output + outputSize, output, Activation::call);
}

template<class Activation> void TrainableDenseInstance<Activation>::forward() {
	gemv(pool->weights.data(), outputSize, inputSize, false, 1, input, 0, outBuffer.data());
	std::transform(outBuffer.data(), outBuffer.data() + outputSize, output, Activation::call);
}

template<class Activation> void DenseInstance<Activation>::backward([[maybe_unused]] FLOAT lrate) {
	throw std::logic_error(".backward called on a non-trainable DenseInstance");
}

template<class Activation> void TrainableDenseInstance<Activation>::backward(FLOAT lrate) {
	std::transform(output, output + outputSize, outBuffer.data(), output,
			[this](const FLOAT& nextder, const FLOAT& o) -> FLOAT {
				auto x = nextder * Activation::derivative(o);
				return antinan(x, antinanFactor);
			});
	ger(-lrate, deltas.data(), outputSize, inputSize, output, input);
	gemv(pool->weights.data(), outputSize, inputSize, true, 1, output, 0, input);
}

template<class Activation> void DenseInstance<Activation>::updateWeights([[maybe_unused]] FLOAT proportion) {
	throw std::logic_error(".updateWeights called on a non-trainable DenseInstance");
}

template<class Activation> void TrainableDenseInstance<Activation>::updateWeights(FLOAT proportion) {
	clamp_axpy(proportion, deltas.data(), deltas.size(), pool->weights.data());
	std::fill(deltas.begin(), deltas.end(), 0);
}
}
