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
#ifndef SEIMEI_NNET_LAYERS_LSTM_HPP_
#include "nnet/layers/lstm.hpp"
#endif

#include "blas.hpp"
#include "nnet/activations.hpp"
#include "xoshiropp.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

namespace seimei::nnet::layers {
inline LSTMBlueprint::LSTMBlueprint() :
		LSTMBlueprint("", "", "", "", 0, 0) {
}

inline LSTMBlueprint::LSTMBlueprint(const string& state_activation, const string& gate_activation,
		const string& output_activation) :
		LSTMBlueprint(state_activation, gate_activation, output_activation, "", 0, 0) {
	if(state_activation.empty() || gate_activation.empty() || output_activation.empty()) {
		throw std::logic_error("Trying to create an LSTMBlueprint with empty activation name(s)");
	}
}

inline LSTMBlueprint::LSTMBlueprint(const string& state_activation, const string& gate_activation,
		const string& output_activation, const string& name, size_t input_size, size_t output_size) :
		inputSize(input_size), outputSize(output_size), stateActivation(state_activation), gateActivation(
				gate_activation), outputActivation(output_activation), name(name) {
}

inline unique_ptr<LayerBlueprint> LSTMBlueprint::clone() const {
	return unique_ptr<LSTMBlueprint>(
			new LSTMBlueprint(stateActivation, gateActivation, outputActivation, name, inputSize, outputSize));
}

inline unique_ptr<LayerBlueprint> LSTMBlueprint::makeShaped(const string& name, size_t input_size,
		size_t output_size) const {
	if(stateActivation.empty()) {
		throw std::logic_error("Trying to shape a default-initialized LSTMBlueprint");
	}
	assertNonZeroShape(name, input_size, output_size);
	return unique_ptr<LSTMBlueprint>(
			new LSTMBlueprint(stateActivation, gateActivation, outputActivation, name, input_size, output_size));
}

inline bool LSTMBlueprint::getShape(string& name, size_t& input_size, size_t& output_size) const {
	if(!inputSize) {
		return false;
	} else {
		name = this->name;
		input_size = inputSize;
		output_size = outputSize;
		return true;
	}
}

inline void LSTMBlueprint::getMemoryRequirements(LayerMemoryRequirements& dest) const {
	if(!inputSize) {
		throw std::logic_error("Called getMemoryRequirements on an LSTMBlueprint without shape");
	}
	// FIXME: revisit once we implement forward-only instances
	dest.numTempStateForward = 0;
	// This should remain, though
	dest.numTempStateBackward = 0;
	dest.szPersistent = 4 * (outputSize * inputSize + outputSize * outputSize) * sizeof(FLOAT);
	// A trainable instance will also have 11 output-sized buffers
	dest.szDeltas = dest.szPersistent * 2 + 11 * outputSize * sizeof(FLOAT);
	// We store the previous state and the output
	dest.szInternalState = outputSize * 2 * sizeof(FLOAT);
}

inline void LSTMBlueprint::serialize([[maybe_unused]] nlohmann::json& out) {
	throw std::logic_error("Serialization of LSTMBlueprint not implemented yet");
}

inline unique_ptr<LayerBlueprint> LSTMBlueprint::deserialize([[maybe_unused]] const nlohmann::json& from) {
	throw std::logic_error("Serialization of LSTMBlueprint not implemented yet");
}

inline shared_ptr<LayerPool> LSTMBlueprint::createPool(const string& network_name, H5::Group* weights_from) const {
	if(!inputSize) {
		throw std::logic_error("Called createPool on an LSTMBlueprint without shape");
	}
	if(weights_from) {
		throw std::logic_error("Loading weights is not yet implemented for LSTM");
	}
	return shared_ptr<LayerPool>(
			new LSTMPool(stateActivation, gateActivation, outputActivation, name, network_name, inputSize, outputSize));
}

inline LSTMPool::LSTMPool(const string& state_activation, const string& gate_activation,
		const string& output_activation, const string& name, const string& network_name, size_t input_size,
		size_t output_size) :
		inputSize(input_size), outputSize(output_size), stateActivation(state_activation), gateActivation(
				gate_activation), outputActivation(output_activation), name(name), networkName(network_name) {
	std::array<vector<FLOAT>*, 4> W {&Winput, &Wforget, &Woutput, &Wstate};
	for(auto p : W) {
		p->resize(outputSize * inputSize);
	}
	std::array<vector<FLOAT>*, 4> U {&Uinput, &Uforget, &Uoutput, &Ustate};
	for(auto p : U) {
		p->resize(outputSize * outputSize);
	}
}

inline const string& LSTMPool::getName() const {
	return name;
}

inline void LSTMPool::initializeWeights(array<uint64_t, 4>& state) {
	std::array<vector<FLOAT>*, 8> weights {&Winput, &Wforget, &Woutput, &Wstate, &Uinput, &Uforget, &Uoutput, &Ustate, };
	for(auto p : weights) {
		for(FLOAT& w : *p) {
			w = from_int(xoshiropp(state)) - 0.5f;
		}
	}
}

inline void LSTMPool::copyWeights([[maybe_unused]] const LayerPool& from) {
	throw std::logic_error("Copying LSTM weights not implemented yet");
}

inline void LSTMPool::saveWeights([[maybe_unused]] H5::Group* to) const {
	throw std::logic_error("Saving LSTM weights not implemented yet");
}

inline unique_ptr<LayerInstance> LSTMPool::createInstance(bool trainable, FLOAT* input, FLOAT* output, FLOAT* temp) {
	auto result = createInstance1(trainable, input, output, temp);
	return unique_ptr<LayerInstance>(result);
}

inline LayerInstance* LSTMPool::createInstance1(bool trainable, FLOAT* input, FLOAT* output, FLOAT* temp) {
	if(stateActivation == "identity") {
		return createInstance2<activations::Identity>(trainable, input, output, temp);
	} else if(stateActivation == "tanh") {
		return createInstance2<activations::TanH>(trainable, input, output, temp);
	} else if(stateActivation == "tanh/uz") {
		return createInstance2<activations::Unzero<activations::TanH>>(trainable, input, output, temp);
	} else if(stateActivation == "sigmoid") {
		return createInstance2<activations::Sigmoid>(trainable, input, output, temp);
	} else {
		throw std::runtime_error(string("Activation not available for LSTM layers (or at all): ") + stateActivation);
	}
}

template<class StateActivation>
LayerInstance* LSTMPool::createInstance2(bool trainable, FLOAT* input, FLOAT* output, FLOAT* temp) {
	if(gateActivation == "identity") {
		return createInstance3<StateActivation, activations::Identity>(trainable, input, output, temp);
	} else if(gateActivation == "tanh") {
		return createInstance3<StateActivation, activations::TanH>(trainable, input, output, temp);
	} else if(gateActivation == "tanh/uz") {
		return createInstance3<StateActivation, activations::Unzero<activations::TanH>>(trainable, input, output, temp);
	} else if(gateActivation == "sigmoid") {
		return createInstance3<StateActivation, activations::Sigmoid>(trainable, input, output, temp);
	} else {
		throw std::runtime_error(string("Activation not available for LSTM layers (or at all): ") + gateActivation);
	}
}

template<class StateActivation, class GateActivation>
LayerInstance* LSTMPool::createInstance3(bool trainable, FLOAT* input, FLOAT* output, [[maybe_unused]] FLOAT* temp) {
	if(!trainable) {
		throw std::logic_error("Non-trainable LSTM instances not implemented yet");
	} else {
		if(outputActivation == "identity") {
			return new TrainableLSTMInstance<StateActivation, GateActivation, activations::Identity>(shared_from_this(),
					input, output);
		} else if(outputActivation == "tanh") {
			return new TrainableLSTMInstance<StateActivation, GateActivation, activations::TanH>(shared_from_this(),
					input, output);
		} else if(outputActivation == "tanh/uz") {
			return new TrainableLSTMInstance<StateActivation, GateActivation, activations::Unzero<activations::TanH>>(
					shared_from_this(), input, output);
		} else if(outputActivation == "sigmoid") {
			return new TrainableLSTMInstance<StateActivation, GateActivation, activations::Sigmoid>(shared_from_this(),
					input, output);
		} else {
			throw std::runtime_error(
					string("Activation not available for LSTM layers (or at all): ") + outputActivation);
		}
	}
}
template<class StateActivation, class GateActivation, class OutputActivation>
TrainableLSTMInstance<StateActivation, GateActivation, OutputActivation>::TrainableLSTMInstance(
		shared_ptr<LSTMPool> pool, FLOAT* input, FLOAT* output) :
		pool(std::move(pool)), input(input), output(output) {
	std::array<vector<FLOAT>*, 4> dW {&dWinput, &dWforget, &dWoutput, &dWstate};
	for(auto p : dW) {
		p->resize(this->pool->outputSize * this->pool->inputSize);
	}
	std::array<vector<FLOAT>*, 4> dU {&dUinput, &dUforget, &dUoutput, &dUstate};
	for(auto p : dU) {
		p->resize(this->pool->outputSize * this->pool->outputSize);
	}
	std::array<vector<FLOAT>*, 13> buffers {&IGBuffer, &inputGate, &FGBuffer, &forgetGate, &OGBuffer, &outputGate,
			&stateBuffer, &activatedStateBuffer, &output1, &output2, &state1, &state2, &activatedOutBuffer, };
	for(auto p : buffers) {
		p->resize(this->pool->outputSize);
	}
}

template<class StateActivation, class GateActivation, class OutputActivation>
void TrainableLSTMInstance<StateActivation, GateActivation, OutputActivation>::forward() {
	/*
	 * The idea:
	 *
	 * output = output_gate * OutputActivation(state)
	 * state = forget_gate * previous_state + input_gate * StateActivation(Wstate @ input + Ustate @ previous_output)
	 * NAME_gate = GateActivation(WNAME @ input + UNAME @ previous_output),
	 *     where NAME is `input`, `forget` or ``output`
	 *
	 * `*` is the Hadamard product, `@` is the matrix-vector multiplication
	 * previous_state an previous_output are state and output after the previous
	 * forward run, zeros after resetState.
	 */
	auto& stored_output = (outputSwitch ? output2 : output1);
	auto& new_output = (outputSwitch ? output1 : output2);
	auto& stored_state = (outputSwitch ? state2 : state1);
	auto& new_state = (outputSwitch ? state1 : state2);
	outputSwitch = !outputSwitch;

	std::array<vector<FLOAT>*, 12> gates {&pool->Winput, &pool->Uinput, &IGBuffer, &inputGate, &pool->Wforget,
			&pool->Uforget, &FGBuffer, &forgetGate, &pool->Woutput, &pool->Uoutput, &OGBuffer, &outputGate, };

	for(size_t i {0}; i < gates.size(); i += 4) {
		gemv(gates[i]->data(), pool->outputSize, pool->inputSize, false, 1, input, 0, gates[i + 2]->data());
		gemv(gates[i + 1]->data(), pool->outputSize, pool->outputSize, false, 1, stored_output.data(), 1,
				gates[i + 2]->data());
		std::transform(gates[i + 2]->begin(), gates[i + 2]->end(), gates[i + 3]->begin(), GateActivation::call);
	}

	gemv(pool->Wstate.data(), pool->outputSize, pool->inputSize, false, 1, input, 0, stateBuffer.data());
	gemv(pool->Ustate.data(), pool->outputSize, pool->outputSize, false, 1, stored_output.data(), 1,
			stateBuffer.data());
	std::transform(stateBuffer.begin(), stateBuffer.end(), activatedStateBuffer.begin(), StateActivation::call);

	for(size_t i {0}; i < pool->outputSize; ++i) {
		FLOAT x {forgetGate[i] * stored_state[i] + inputGate[i] * activatedStateBuffer[i]};
		new_state[i] = x;
		x = OutputActivation::call(x);
		activatedOutBuffer[i] = x;
		x *= outputGate[i];
		new_output[i] = x;
		output[i] = x;
	}
}

template<class StateActivation, class GateActivation, class OutputActivation>
void TrainableLSTMInstance<StateActivation, GateActivation, OutputActivation>::backward(FLOAT lrate) {
	// We do not actually unroll the network. That would require more memory than I (Fedor) have
	// and I'm not sure there is any point in doing so in Seimei's particular case

	// WARNING: this is not guaranteed to be correct :(

	// Output switch will have been flipped by forward, so mirror it
	auto& stored_output = (!outputSwitch ? output2 : output1);
	auto& stored_state = (!outputSwitch ? state2 : state1);
	auto& new_state = (!outputSwitch ? state1 : state2);

	// Updating the output gate and simultaneously propagating the error through the output function
	for(size_t i {0}; i < pool->outputSize; ++i) {
		// Output gate
		OGBuffer[i] = antinan(output[i] * activatedOutBuffer[i] * GateActivation::derivative(OGBuffer[i]),
				antinanFactor);
		// Propagating through output function
		FLOAT x {output[i] * outputGate[i] * OutputActivation::derivative(new_state[i])};
		// Forget gate
		FGBuffer[i] = antinan(x * stored_state[i] * GateActivation::derivative(FGBuffer[i]), antinanFactor);
		// Input gate
		IGBuffer[i] = antinan(x * activatedStateBuffer[i] * GateActivation::derivative(IGBuffer[i]), antinanFactor);
		// State weights
		stateBuffer[i] = antinan(x * inputGate[i] * StateActivation::derivative(stateBuffer[i]), antinanFactor);
	}
	// Update weights and propagate error
	// @formatter:off
	std::array<FLOAT*, 16> updates {
		IGBuffer.data(), dWinput.data(), dUinput.data(), pool->Winput.data(),
		FGBuffer.data(), dWforget.data(), dUforget.data(), pool->Wforget.data(),
		OGBuffer.data(), dWoutput.data(), dUoutput.data(), pool->Woutput.data(),
		stateBuffer.data(), dWstate.data(), dUstate.data(), pool->Wstate.data(),
	};
	// @formatter:on
	for(size_t i {0}; i < updates.size(); i += 4) {
		ger(-lrate, updates[i + 1], pool->outputSize, pool->inputSize, updates[i], input);
		ger(-lrate, updates[i + 2], pool->outputSize, pool->outputSize, updates[i], stored_output.data());
		gemv(updates[i + 3], pool->outputSize, pool->inputSize, true, 1, updates[i], (i > 0), input);
	}
}

template<class StateActivation, class GateActivation, class OutputActivation>
void TrainableLSTMInstance<StateActivation, GateActivation, OutputActivation>::updateWeights(FLOAT proportion) {
	// @formatter:off
	std::array<vector<FLOAT>*, 16> weights {
		&dWinput, &pool->Winput, &dUinput, &pool->Uinput,
		&dWforget, &pool->Wforget, &dUforget, &pool->Uforget,
		&dWoutput, &pool->Woutput, &dUoutput, &pool->Uoutput,
		&dWstate, &pool->Wstate, &dUstate, &pool->Ustate,
	};
	// @formatter:on
	for(size_t i {0}; i < weights.size(); i += 2) {
		clamp_axpy(proportion, weights.at(i)->data(), weights.at(i)->size(), weights.at(i + 1)->data());
		std::fill(weights.at(i)->begin(), weights.at(i)->end(), 0);
	}
}

template<class StateActivation, class GateActivation, class OutputActivation>
void TrainableLSTMInstance<StateActivation, GateActivation, OutputActivation>::resetState() {
	std::array<vector<FLOAT>*, 4> to_reset {&output1, &output2, &state1, &state2};
	for(auto p : to_reset) {
		std::fill(p->begin(), p->end(), 0);
	}
}
}
