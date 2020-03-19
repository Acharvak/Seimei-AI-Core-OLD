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

/// An LSTM layer

#ifndef SEIMEI_NNET_LAYERS_LSTM_HPP_
#define SEIMEI_NNET_LAYERS_LSTM_HPP_

#include "common.hpp"
#include "nnet/framework.hpp"

namespace seimei::nnet::layers {
class LSTMBlueprint : public LayerBlueprint {
public:
	LSTMBlueprint();

	LSTMBlueprint(
			const string& state_activation,
			const string& gate_activation,
			const string& output_activation);

	virtual ~LSTMBlueprint() = default;

	virtual unique_ptr<LayerBlueprint> clone() const override;

	virtual unique_ptr<LayerBlueprint> makeShaped(
				const string& name,
				size_t input_size, size_t output_size) const override;

	virtual bool getShape(string& name,
				size_t& input_size, size_t& output_size) const override;

	virtual void getMemoryRequirements(LayerMemoryRequirements& dest) const override;

	virtual void serialize(nlohmann::json& out) override;

	virtual unique_ptr<LayerBlueprint> deserialize(const nlohmann::json& from) override;

	virtual shared_ptr<class LayerPool> createPool(
				const string& network_name,
				H5::Group * weights_from) const override;
protected:
	size_t inputSize, outputSize;
	string stateActivation, gateActivation, outputActivation, name;

	LSTMBlueprint(const string& state_activation, const string& gate_activation, const string& output_activation,
			const string& name, size_t input_size, size_t output_size);
};

class LSTMPool : public LayerPool, public std::enable_shared_from_this<LSTMPool> {
	friend LSTMBlueprint;
	template<class StateActivation, class GateActivation, class OutputActivation>
	friend class LSTMInstance;
	template<class StateActivation, class GateActivation, class OutputActivation>
	friend class TrainableLSTMInstance;
public:
	virtual const string& getName() const override;
	virtual void initializeWeights(array<uint64_t, 4>& state) override;
	virtual void copyWeights(const LayerPool& from) override;
	virtual void saveWeights(H5::Group * to) const override;
	virtual unique_ptr<LayerInstance> createInstance(
				bool trainable,
				FLOAT * input, FLOAT * output, FLOAT * temp) override;
protected:
	// W-weights are for the input, U-weights are for the previous output
	const size_t inputSize, outputSize;
	const string stateActivation, gateActivation, outputActivation, name, networkName;
	vector<FLOAT> Winput {}, Wforget {}, Woutput {}, Wstate {};
	vector<FLOAT> Uinput {}, Uforget {}, Uoutput {}, Ustate {};

	LSTMPool(const string& state_activation, const string& gate_activation, const string& output_activation,
			const string& name, const string& network_name, size_t input_size, size_t output_size);

	LayerInstance* createInstance1(bool trainable, FLOAT * input, FLOAT * output, FLOAT * temp);
	template<class StateActivation> LayerInstance* createInstance2(bool trainable, FLOAT * input, FLOAT * output, FLOAT * temp);
	template<class StateActivation, class GateActivation> LayerInstance* createInstance3(bool trainable, FLOAT * input, FLOAT * output, FLOAT * temp);
};

/*
template<class StateActivation, class GateActivation, class OutputActivation>
class LSTMInstance : public LayerInstance {
	friend LSTMPool;
public:
	virtual ~LSTMInstance() = default;

	virtual void forward();
	virtual void backward(FLOAT lrate);
	virtual void updateWeights(FLOAT proportion);
	virtual void resetState();
};
*/

template<class StateActivation, class GateActivation, class OutputActivation>
class TrainableLSTMInstance : public LayerInstance {
	friend LSTMPool;
public:
	virtual ~TrainableLSTMInstance() = default;

	virtual void forward();
	virtual void backward(FLOAT lrate);
	virtual void updateWeights(FLOAT proportion);
	virtual void resetState();
protected:
	bool outputSwitch {true};
	FLOAT antinanFactor {DEFAULT_ANTINAN_FACTOR};
	shared_ptr<LSTMPool> pool;
	FLOAT* input, * output;
	vector<FLOAT> dWinput {}, dWforget {}, dWoutput {}, dWstate {};
	vector<FLOAT> dUinput {}, dUforget {}, dUoutput {}, dUstate {};
	vector<FLOAT> IGBuffer {}, inputGate {}, FGBuffer {}, forgetGate {},
		OGBuffer {}, outputGate {}, stateBuffer {}, activatedStateBuffer {},
		output1 {}, output2 {}, state1 {}, state2 {}, activatedOutBuffer {};
	TrainableLSTMInstance(shared_ptr<LSTMPool> pool, FLOAT* input, FLOAT* output);
};
}

#include "nnet/layers/lstm.tpp"
#endif
