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

#ifndef SEIMEI_NNET_LAYERS_DENSE_HPP_
#define SEIMEI_NNET_LAYERS_DENSE_HPP_

#include "common.hpp"
#include "nnet/framework.hpp"

/**
 * This is header-only (because everything that's not a template
 * class is rather simple anyway).
 */

namespace seimei::nnet::layers {
/**
 * A fully connected layer.
 */
class DenseBlueprint : public LayerBlueprint {
public:
	DenseBlueprint();
	DenseBlueprint(const string& activation);
	virtual ~DenseBlueprint() = default;

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
	const string name, activation;
	const size_t inputSize, outputSize;

	DenseBlueprint(const string& name,
			const string& activation,
			size_t input_size, size_t output_size);
};


class DensePool : public LayerPool, public std::enable_shared_from_this<DensePool> {
	friend DenseBlueprint;
	template<class Activation> friend class DenseInstance;
	template<class Activation> friend class TrainableDenseInstance;
public:
	virtual const string& getName() const override;
	virtual void initializeWeights(array<uint64_t, 4>& state) override;
	virtual void copyWeights(const LayerPool& from) override;
	virtual void saveWeights(H5::Group * to) const override;
	virtual unique_ptr<LayerInstance> createInstance(
				bool trainable,
				FLOAT * input, FLOAT * output, FLOAT * temp) override;
protected:
	const size_t inputSize, outputSize;
	const string name, networkName, activation;
	vector<FLOAT> weights;

	DensePool(const string& name,
			const string& network_name,
			const string& activation,
			size_t input_size, size_t output_size);
	void loadWeights(H5::Group * weights_from);
	template<class Activation> LayerInstance * createRawInstance(
			bool trainable, FLOAT * input, FLOAT * output);
};


template<class Activation>
class DenseInstance : public LayerInstance {
	friend DensePool;
public:
	DenseInstance(DenseInstance&) = delete;
	DenseInstance(DenseInstance&&) = delete;
	DenseInstance& operator=(DenseInstance&) = delete;
	DenseInstance& operator=(DenseInstance&&) = delete;
	virtual ~DenseInstance() = default;

	virtual void forward();
	virtual void backward(FLOAT lrate);
	virtual void updateWeights(FLOAT proportion);
protected:
	shared_ptr<DensePool> pool;
	FLOAT *input, *output;
	size_t inputSize, outputSize;

	DenseInstance(shared_ptr<DensePool> pool,
			FLOAT * input, FLOAT * output,
			size_t input_size, size_t output_size);
};


template<class Activation>
class TrainableDenseInstance : public DenseInstance<Activation> {
	friend DensePool;
public:
	virtual ~TrainableDenseInstance() = default;

	virtual void forward();
	virtual void backward(FLOAT lrate);
	virtual void updateWeights(FLOAT proportion);
protected:
	using DenseInstance<Activation>::pool;
	using DenseInstance<Activation>::input;
	using DenseInstance<Activation>::output;
	using DenseInstance<Activation>::inputSize;
	using DenseInstance<Activation>::outputSize;
	vector<FLOAT> deltas, outBuffer;
	FLOAT antinanFactor {DEFAULT_ANTINAN_FACTOR};

	TrainableDenseInstance(shared_ptr<DensePool> pool,
			FLOAT * input, FLOAT * output,
			size_t input_size, size_t output_size);
};
}
#include "nnet/layers/dense.tpp"
#endif
