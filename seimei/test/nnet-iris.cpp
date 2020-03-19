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
 * Tests that train neural networks on the iris dataset.
 */

#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "common.hpp"
#include "xoshiropp.hpp"
#include "nnet/framework.hpp"
#include "nnet/layers.hpp"

/*
 * This is R.A. Fisher's iris dataset, as published under
 *     https://archive.ics.uci.edu/ml/datasets/Iris/
 *
 * The 35th and 38th samples have been corrected as documented
 * on the page.
 */

const FLOAT IRIS_SETOSA {0};
const FLOAT IRIS_VERSICOLOR {1};
const FLOAT IRIS_VIRGINICA {2};

const size_t DSENTRY_SIZE {5};

const std::vector<FLOAT> dataset {
	5.1, 3.5, 1.4, 0.2, IRIS_SETOSA,
	4.9, 3.0, 1.4, 0.2, IRIS_SETOSA,
	4.7, 3.2, 1.3, 0.2, IRIS_SETOSA,
	4.6, 3.1, 1.5, 0.2, IRIS_SETOSA,
	5.0, 3.6, 1.4, 0.2, IRIS_SETOSA,
	5.4, 3.9, 1.7, 0.4, IRIS_SETOSA,
	4.6, 3.4, 1.4, 0.3, IRIS_SETOSA,
	5.0, 3.4, 1.5, 0.2, IRIS_SETOSA,
	4.4, 2.9, 1.4, 0.2, IRIS_SETOSA,
	4.9, 3.1, 1.5, 0.1, IRIS_SETOSA,
	5.4, 3.7, 1.5, 0.2, IRIS_SETOSA,
	4.8, 3.4, 1.6, 0.2, IRIS_SETOSA,
	4.8, 3.0, 1.4, 0.1, IRIS_SETOSA,
	4.3, 3.0, 1.1, 0.1, IRIS_SETOSA,
	5.8, 4.0, 1.2, 0.2, IRIS_SETOSA,
	5.7, 4.4, 1.5, 0.4, IRIS_SETOSA,
	5.4, 3.9, 1.3, 0.4, IRIS_SETOSA,
	5.1, 3.5, 1.4, 0.3, IRIS_SETOSA,
	5.7, 3.8, 1.7, 0.3, IRIS_SETOSA,
	5.1, 3.8, 1.5, 0.3, IRIS_SETOSA,
	5.4, 3.4, 1.7, 0.2, IRIS_SETOSA,
	5.1, 3.7, 1.5, 0.4, IRIS_SETOSA,
	4.6, 3.6, 1.0, 0.2, IRIS_SETOSA,
	5.1, 3.3, 1.7, 0.5, IRIS_SETOSA,
	4.8, 3.4, 1.9, 0.2, IRIS_SETOSA,
	5.0, 3.0, 1.6, 0.2, IRIS_SETOSA,
	5.0, 3.4, 1.6, 0.4, IRIS_SETOSA,
	5.2, 3.5, 1.5, 0.2, IRIS_SETOSA,
	5.2, 3.4, 1.4, 0.2, IRIS_SETOSA,
	4.7, 3.2, 1.6, 0.2, IRIS_SETOSA,
	4.8, 3.1, 1.6, 0.2, IRIS_SETOSA,
	5.4, 3.4, 1.5, 0.4, IRIS_SETOSA,
	5.2, 4.1, 1.5, 0.1, IRIS_SETOSA,
	5.5, 4.2, 1.4, 0.2, IRIS_SETOSA,
	4.9, 3.1, 1.5, 0.2, IRIS_SETOSA,
	5.0, 3.2, 1.2, 0.2, IRIS_SETOSA,
	5.5, 3.5, 1.3, 0.2, IRIS_SETOSA,
	4.9, 3.6, 1.4, 0.1, IRIS_SETOSA,
	4.4, 3.0, 1.3, 0.2, IRIS_SETOSA,
	5.1, 3.4, 1.5, 0.2, IRIS_SETOSA,
	5.0, 3.5, 1.3, 0.3, IRIS_SETOSA,
	4.5, 2.3, 1.3, 0.3, IRIS_SETOSA,
	4.4, 3.2, 1.3, 0.2, IRIS_SETOSA,
	5.0, 3.5, 1.6, 0.6, IRIS_SETOSA,
	5.1, 3.8, 1.9, 0.4, IRIS_SETOSA,
	4.8, 3.0, 1.4, 0.3, IRIS_SETOSA,
	5.1, 3.8, 1.6, 0.2, IRIS_SETOSA,
	4.6, 3.2, 1.4, 0.2, IRIS_SETOSA,
	5.3, 3.7, 1.5, 0.2, IRIS_SETOSA,
	5.0, 3.3, 1.4, 0.2, IRIS_SETOSA,
	7.0, 3.2, 4.7, 1.4, IRIS_VERSICOLOR,
	6.4, 3.2, 4.5, 1.5, IRIS_VERSICOLOR,
	6.9, 3.1, 4.9, 1.5, IRIS_VERSICOLOR,
	5.5, 2.3, 4.0, 1.3, IRIS_VERSICOLOR,
	6.5, 2.8, 4.6, 1.5, IRIS_VERSICOLOR,
	5.7, 2.8, 4.5, 1.3, IRIS_VERSICOLOR,
	6.3, 3.3, 4.7, 1.6, IRIS_VERSICOLOR,
	4.9, 2.4, 3.3, 1.0, IRIS_VERSICOLOR,
	6.6, 2.9, 4.6, 1.3, IRIS_VERSICOLOR,
	5.2, 2.7, 3.9, 1.4, IRIS_VERSICOLOR,
	5.0, 2.0, 3.5, 1.0, IRIS_VERSICOLOR,
	5.9, 3.0, 4.2, 1.5, IRIS_VERSICOLOR,
	6.0, 2.2, 4.0, 1.0, IRIS_VERSICOLOR,
	6.1, 2.9, 4.7, 1.4, IRIS_VERSICOLOR,
	5.6, 2.9, 3.6, 1.3, IRIS_VERSICOLOR,
	6.7, 3.1, 4.4, 1.4, IRIS_VERSICOLOR,
	5.6, 3.0, 4.5, 1.5, IRIS_VERSICOLOR,
	5.8, 2.7, 4.1, 1.0, IRIS_VERSICOLOR,
	6.2, 2.2, 4.5, 1.5, IRIS_VERSICOLOR,
	5.6, 2.5, 3.9, 1.1, IRIS_VERSICOLOR,
	5.9, 3.2, 4.8, 1.8, IRIS_VERSICOLOR,
	6.1, 2.8, 4.0, 1.3, IRIS_VERSICOLOR,
	6.3, 2.5, 4.9, 1.5, IRIS_VERSICOLOR,
	6.1, 2.8, 4.7, 1.2, IRIS_VERSICOLOR,
	6.4, 2.9, 4.3, 1.3, IRIS_VERSICOLOR,
	6.6, 3.0, 4.4, 1.4, IRIS_VERSICOLOR,
	6.8, 2.8, 4.8, 1.4, IRIS_VERSICOLOR,
	6.7, 3.0, 5.0, 1.7, IRIS_VERSICOLOR,
	6.0, 2.9, 4.5, 1.5, IRIS_VERSICOLOR,
	5.7, 2.6, 3.5, 1.0, IRIS_VERSICOLOR,
	5.5, 2.4, 3.8, 1.1, IRIS_VERSICOLOR,
	5.5, 2.4, 3.7, 1.0, IRIS_VERSICOLOR,
	5.8, 2.7, 3.9, 1.2, IRIS_VERSICOLOR,
	6.0, 2.7, 5.1, 1.6, IRIS_VERSICOLOR,
	5.4, 3.0, 4.5, 1.5, IRIS_VERSICOLOR,
	6.0, 3.4, 4.5, 1.6, IRIS_VERSICOLOR,
	6.7, 3.1, 4.7, 1.5, IRIS_VERSICOLOR,
	6.3, 2.3, 4.4, 1.3, IRIS_VERSICOLOR,
	5.6, 3.0, 4.1, 1.3, IRIS_VERSICOLOR,
	5.5, 2.5, 4.0, 1.3, IRIS_VERSICOLOR,
	5.5, 2.6, 4.4, 1.2, IRIS_VERSICOLOR,
	6.1, 3.0, 4.6, 1.4, IRIS_VERSICOLOR,
	5.8, 2.6, 4.0, 1.2, IRIS_VERSICOLOR,
	5.0, 2.3, 3.3, 1.0, IRIS_VERSICOLOR,
	5.6, 2.7, 4.2, 1.3, IRIS_VERSICOLOR,
	5.7, 3.0, 4.2, 1.2, IRIS_VERSICOLOR,
	5.7, 2.9, 4.2, 1.3, IRIS_VERSICOLOR,
	6.2, 2.9, 4.3, 1.3, IRIS_VERSICOLOR,
	5.1, 2.5, 3.0, 1.1, IRIS_VERSICOLOR,
	5.7, 2.8, 4.1, 1.3, IRIS_VERSICOLOR,
	6.3, 3.3, 6.0, 2.5, IRIS_VIRGINICA,
	5.8, 2.7, 5.1, 1.9, IRIS_VIRGINICA,
	7.1, 3.0, 5.9, 2.1, IRIS_VIRGINICA,
	6.3, 2.9, 5.6, 1.8, IRIS_VIRGINICA,
	6.5, 3.0, 5.8, 2.2, IRIS_VIRGINICA,
	7.6, 3.0, 6.6, 2.1, IRIS_VIRGINICA,
	4.9, 2.5, 4.5, 1.7, IRIS_VIRGINICA,
	7.3, 2.9, 6.3, 1.8, IRIS_VIRGINICA,
	6.7, 2.5, 5.8, 1.8, IRIS_VIRGINICA,
	7.2, 3.6, 6.1, 2.5, IRIS_VIRGINICA,
	6.5, 3.2, 5.1, 2.0, IRIS_VIRGINICA,
	6.4, 2.7, 5.3, 1.9, IRIS_VIRGINICA,
	6.8, 3.0, 5.5, 2.1, IRIS_VIRGINICA,
	5.7, 2.5, 5.0, 2.0, IRIS_VIRGINICA,
	5.8, 2.8, 5.1, 2.4, IRIS_VIRGINICA,
	6.4, 3.2, 5.3, 2.3, IRIS_VIRGINICA,
	6.5, 3.0, 5.5, 1.8, IRIS_VIRGINICA,
	7.7, 3.8, 6.7, 2.2, IRIS_VIRGINICA,
	7.7, 2.6, 6.9, 2.3, IRIS_VIRGINICA,
	6.0, 2.2, 5.0, 1.5, IRIS_VIRGINICA,
	6.9, 3.2, 5.7, 2.3, IRIS_VIRGINICA,
	5.6, 2.8, 4.9, 2.0, IRIS_VIRGINICA,
	7.7, 2.8, 6.7, 2.0, IRIS_VIRGINICA,
	6.3, 2.7, 4.9, 1.8, IRIS_VIRGINICA,
	6.7, 3.3, 5.7, 2.1, IRIS_VIRGINICA,
	7.2, 3.2, 6.0, 1.8, IRIS_VIRGINICA,
	6.2, 2.8, 4.8, 1.8, IRIS_VIRGINICA,
	6.1, 3.0, 4.9, 1.8, IRIS_VIRGINICA,
	6.4, 2.8, 5.6, 2.1, IRIS_VIRGINICA,
	7.2, 3.0, 5.8, 1.6, IRIS_VIRGINICA,
	7.4, 2.8, 6.1, 1.9, IRIS_VIRGINICA,
	7.9, 3.8, 6.4, 2.0, IRIS_VIRGINICA,
	6.4, 2.8, 5.6, 2.2, IRIS_VIRGINICA,
	6.3, 2.8, 5.1, 1.5, IRIS_VIRGINICA,
	6.1, 2.6, 5.6, 1.4, IRIS_VIRGINICA,
	7.7, 3.0, 6.1, 2.3, IRIS_VIRGINICA,
	6.3, 3.4, 5.6, 2.4, IRIS_VIRGINICA,
	6.4, 3.1, 5.5, 1.8, IRIS_VIRGINICA,
	6.0, 3.0, 4.8, 1.8, IRIS_VIRGINICA,
	6.9, 3.1, 5.4, 2.1, IRIS_VIRGINICA,
	6.7, 3.1, 5.6, 2.4, IRIS_VIRGINICA,
	6.9, 3.1, 5.1, 2.3, IRIS_VIRGINICA,
	5.8, 2.7, 5.1, 1.9, IRIS_VIRGINICA,
	6.8, 3.2, 5.9, 2.3, IRIS_VIRGINICA,
	6.7, 3.3, 5.7, 2.5, IRIS_VIRGINICA,
	6.7, 3.0, 5.2, 2.3, IRIS_VIRGINICA,
	6.3, 2.5, 5.0, 1.9, IRIS_VIRGINICA,
	6.5, 3.0, 5.2, 2.0, IRIS_VIRGINICA,
	6.2, 3.4, 5.4, 2.3, IRIS_VIRGINICA,
	5.9, 3.0, 5.1, 1.8, IRIS_VIRGINICA
};

class IrisTests : public ::testing::Test {
public:
	typedef std::array<std::array<size_t, 3>, 3> confmatrix_t;
protected:
	std::vector<FLOAT> train, test;
	std::array<std::uint64_t, 4> rngState {1, 2, 3, 4};

	static void makeShufflingVector(size_t number, std::vector<size_t>& result, std::array<std::uint64_t, 4>& rng_state);
	static void printConfusionMatrix(const confmatrix_t& matrix);

	void resetRNG();
	void SetUp() override;
	// Do one epoch. Will not call .updateWeights
	void trainNetwork(FLOAT lrate, FLOAT correct_out, FLOAT incorrect_out, seimei::nnet::NetworkInstance& nnet);
	// Fake time series by providing every feature in order. Dynamically scale the lrate.
	void trainNetwork_timeSeries(FLOAT lrate, FLOAT correct_out, FLOAT incorrect_out, seimei::nnet::NetworkInstance& nnet);
	// Return accuracy
	FLOAT testNetwork(seimei::nnet::NetworkInstance& nnet, confmatrix_t& confusion_matrix);
	// Will only take the last output of the network into account
	FLOAT testNetwork_timeSeries(seimei::nnet::NetworkInstance& nnet, confmatrix_t& confusion_matrix);
};

void IrisTests::makeShufflingVector(size_t number, std::vector<size_t>& result, std::array<std::uint64_t, 4>& rng_state) {
	std::vector<size_t> remaining(number);
	std::iota(remaining.begin(), remaining.end(), 0);
	result.resize(number);
	for(size_t i {0}; i < number; ++i) {
		size_t index {static_cast<size_t>(seimei::xoshiropp(rng_state) % static_cast<uint64_t>(remaining.size()))};
		result.at(i) = remaining.at(index);
		remaining.erase(remaining.begin() + index);
	}
}

void IrisTests::printConfusionMatrix(const confmatrix_t& matrix) {
	std::cerr << "CONFUSION MATRIX (Expected/Got: Setosa, Versicolor, Virginica)" << std::endl;
	for(size_t row {0}; row < 3; ++row) {
		std::cerr << matrix.at(row).at(0) << " " << matrix.at(row).at(1) << " " << matrix.at(row).at(2) << std::endl;
	}
}

void IrisTests::resetRNG() {
	std::iota(rngState.begin(), rngState.end(), 1);
}

void IrisTests::SetUp() {
	// The first 5 of each class go into test, the rest go into train
	const size_t TEST_SPLIT {5};
	FLOAT last_class {100};
	size_t last_class_test {0};
	for(size_t i {0}; i < dataset.size(); i += DSENTRY_SIZE) {
		if(dataset.at(i + 4) != last_class) {
			last_class = dataset.at(i + DSENTRY_SIZE - 1);
			last_class_test = 0;
		}
		// Not the most efficient way, but will suffice for a test
		for(size_t k {i}; k < i + DSENTRY_SIZE; ++k) {
			if(last_class_test < TEST_SPLIT) {
				test.push_back(dataset.at(k));
			} else {
				train.push_back(dataset.at(k));
			}
		}
		++last_class_test;
	}
}

void IrisTests::trainNetwork(FLOAT lrate, FLOAT correct_out, FLOAT incorrect_out, seimei::nnet::NetworkInstance& nnet) {
	ASSERT_GT(correct_out, incorrect_out);
	std::vector<size_t> indices;
	makeShufflingVector(train.size() / DSENTRY_SIZE, indices, rngState);
	FLOAT* input, * output;
	size_t input_size {nnet.getInput(&input)};
	ASSERT_EQ(input_size, DSENTRY_SIZE - 1);
	size_t output_size {nnet.getOutput(&output)};
	ASSERT_EQ(output_size, 3);
	std::vector<FLOAT> expected(3);

	for(size_t i : indices) {
		std::copy(train.data() + i * DSENTRY_SIZE, train.data() + i * DSENTRY_SIZE + DSENTRY_SIZE - 1, input);
		nnet.forward();
		std::fill(expected.begin(), expected.end(), incorrect_out);
		expected.at(static_cast<size_t>(train.at(i * DSENTRY_SIZE + DSENTRY_SIZE - 1))) = correct_out;
		nnet.backward(lrate, expected.data());
	}
}

void IrisTests::trainNetwork_timeSeries(FLOAT lrate, FLOAT correct_out, FLOAT incorrect_out, seimei::nnet::NetworkInstance& nnet) {
	ASSERT_GT(correct_out, incorrect_out);
	std::vector<size_t> indices;
	makeShufflingVector(train.size() / DSENTRY_SIZE, indices, rngState);
	FLOAT* input, * output;
	size_t input_size {nnet.getInput(&input)};
	ASSERT_EQ(input_size, 1);
	size_t output_size {nnet.getOutput(&output)};
	ASSERT_EQ(output_size, 3);
	std::vector<FLOAT> expected(3);

	for(size_t i : indices) {
		nnet.resetState();
		for(size_t k {0}; k < DSENTRY_SIZE - 1; ++k) {
			*input = train.at(i * DSENTRY_SIZE + k);
			nnet.forward();
			std::fill(expected.begin(), expected.end(), incorrect_out);
			expected.at(static_cast<size_t>(train.at(i * DSENTRY_SIZE + DSENTRY_SIZE - 1))) = correct_out;
			FLOAT lrate_mod {static_cast<FLOAT>(k) / (DSENTRY_SIZE - 2)};
			lrate_mod *= lrate_mod;
			nnet.backward(lrate * lrate_mod, expected.data());
			//nnet.backward(lrate, expected.data());
		}
	}
}

FLOAT IrisTests::testNetwork(seimei::nnet::NetworkInstance& nnet, confmatrix_t& confusion_matrix) {
	FLOAT* input, * output;
	size_t input_size {nnet.getInput(&input)};
	// Looks like we can't use Google Test macros in functions returning non-void
	assert(input_size == DSENTRY_SIZE - 1);
	size_t output_size {nnet.getOutput(&output)};
	assert(output_size == 3);
	for(auto& row : confusion_matrix) {
		row.fill(0);
	}

	FLOAT num_good {0};

	for(size_t i {0}; i < test.size(); i += DSENTRY_SIZE) {
		std::copy(test.data() + i, test.data() + i + DSENTRY_SIZE - 1, input);
		nnet.forward();
		auto best = std::max<size_t>({0, 1, 2}, [&](size_t a, size_t b) -> bool {
			return (output[a] < output[b]);
		});
		size_t expected {static_cast<size_t>(test.at(i + DSENTRY_SIZE - 1))};
		++confusion_matrix.at(expected).at(best);
		if(expected == best) {
			++num_good;
		}
	}

	return num_good / static_cast<FLOAT>(test.size() / DSENTRY_SIZE);
}

FLOAT IrisTests::testNetwork_timeSeries(seimei::nnet::NetworkInstance& nnet, confmatrix_t& confusion_matrix) {
	FLOAT* input, * output;
	size_t input_size {nnet.getInput(&input)};
	assert(input_size == 1);
	size_t output_size {nnet.getOutput(&output)};
	assert(output_size == 3);
	for(auto& row : confusion_matrix) {
		row.fill(0);
	}

	FLOAT num_good {0};

	for(size_t i {0}; i < test.size(); i += DSENTRY_SIZE) {
		nnet.resetState();
		for(size_t k {0}; k < DSENTRY_SIZE - 1; ++k) {
			*input = test.at(i + k);
			nnet.forward();
		}
		auto best = std::max<size_t>({0, 1, 2}, [&](size_t a, size_t b) -> bool {
			return (output[a] < output[b]);
		});
		size_t expected {static_cast<size_t>(test.at(i + DSENTRY_SIZE - 1))};
		++confusion_matrix.at(expected).at(best);
		if(expected == best) {
			++num_good;
		}
	}

	return num_good / static_cast<FLOAT>(test.size() / DSENTRY_SIZE);
}

TEST_F(IrisTests, Double_Perceptron) {
	seimei::nnet::NetworkBlueprint nbp("Iris Test - 2 layers", 4);
	//seimei::nnet::layers::ActivationBlueprint input_act("tanh");
	//auto idx = nbp.addLayer(input_act, "Input Activation", 0, false, 4);
	size_t idx {0};
	seimei::nnet::layers::DenseBlueprint input2hidden("tanh");
	idx = nbp.addLayer(input2hidden, "Input to Hidden", idx, false, 500);
	seimei::nnet::layers::DenseBlueprint hidden2output("tanh");
	idx = nbp.addLayer(hidden2output, "Hidden to Output", idx, true, 3);
	nbp.setNetworkOutput(idx);

	auto pool = nbp.createPool(nullptr);
	std::array<std::uint64_t, 4> seed {1, 2, 3, 4};
	pool->initializeWeights(seed);

	auto nnet = pool->createInstance(true);
	// 100 epochs with lrate = 0.0001
	resetRNG();
	for(size_t i {0}; i < 100; ++i) {
		trainNetwork(0.0001f, 1.0f, -1.0f, *nnet);
		nnet->updateWeights(1.0f);
	}
	// Test it
	confmatrix_t confmatrix {};
	auto accuracy = testNetwork(*nnet, confmatrix);
	printConfusionMatrix(confmatrix);
	EXPECT_GT(accuracy, 0.9f);
}

TEST_F(IrisTests, LSTM) {
	seimei::nnet::NetworkBlueprint nbp("Iris Test - LSTM", 1);
	size_t idx {0};
	//seimei::nnet::layers::ActivationBlueprint input_act("tanh");
	//idx = nbp.addLayer(input_act, "Input Activation", idx, false, 1);
	seimei::nnet::layers::LSTMBlueprint lstm("tanh", "sigmoid", "identity");
	idx = nbp.addLayer(lstm, "LSTM Layer 1", idx, true, 300);
	seimei::nnet::layers::DenseBlueprint dense("tanh");
	idx = nbp.addLayer(dense, "Dense Output Layer", idx, false, 3);
	nbp.setNetworkOutput(idx);

	auto pool = nbp.createPool(nullptr);
	std::array<std::uint64_t, 4> seed {1, 2, 3, 4};
	pool->initializeWeights(seed);

	auto nnet = pool->createInstance(true);
	// 200 epochs with lrate = 0.0001
	resetRNG();
	for(size_t i {0}; i < 200; ++i) {
		trainNetwork_timeSeries(0.0001f, 1.0f, -1.0f, *nnet);
		nnet->updateWeights(1.0f);
	}
	// Test it
	confmatrix_t confmatrix {};
	auto accuracy = testNetwork_timeSeries(*nnet, confmatrix);
	printConfusionMatrix(confmatrix);
	EXPECT_GT(accuracy, 0.9f);
}
