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
 * These are neural network tests that are fast and don't require
 * actually training neural networks.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <limits>

#define SEIMEI_TESTER_NNET_FRIEND NNetTestInternal

#include "nnet/activations.hpp"
#include "nnet/framework.hpp"
#include "nnet/layers.hpp"

#ifdef SEIMEI_DOUBLE_PRECISION
#define _ASSERT_FLOAT_EQ ASSERT_FLOAT_EQ
#define _EXPECT_FLOAT_EQ EXPECT_FLOAT_EQ
#else
#define _ASSERT_FLOAT_EQ ASSERT_FLOAT_EQ
#define _EXPECT_FLOAT_EQ EXPECT_FLOAT_EQ
#endif

// ===== FOR INFORMATIONAL PURPOSES =====
#ifdef SEIMEI_DOUBLE_PRECISION
TEST(For_Your_Information, Precision_is_Double) {
}
#else
TEST(For_Your_Information, Precision_is_Single) {
}
#endif

// ===== TEST ACTIVATIONS =====
namespace seimei::nnet::activations {
TEST(Activations, Identity) {
	_EXPECT_FLOAT_EQ(Identity::call(10.0), 10.0);
	_EXPECT_FLOAT_EQ(Identity::derivative(10.0), 1.0);
}

TEST(Activations, TanH) {
	_EXPECT_FLOAT_EQ(TanH::call(7.0), 0.9999983369439446717);
	_EXPECT_FLOAT_EQ(TanH::derivative(7.0), 0.00000332611);
	_EXPECT_FLOAT_EQ(TanH::call(-1.0), -0.7615941559557649);
	_EXPECT_FLOAT_EQ(TanH::derivative(-1.0), 0.4199743416140260693);
	_EXPECT_FLOAT_EQ(TanH::call(0.0), 0.0);
	_EXPECT_FLOAT_EQ(TanH::derivative(0.0), 1.0);
}

TEST(Activations, Sigmoid) {
	_EXPECT_FLOAT_EQ(Sigmoid::call(10.0), 0.99995460213129756560549);
	_EXPECT_FLOAT_EQ(Sigmoid::derivative(10.0), 0.00004539580773595167);
	_EXPECT_FLOAT_EQ(Sigmoid::call(0.0), 0.5);
	_EXPECT_FLOAT_EQ(Sigmoid::derivative(0.0), 0.25);
	_EXPECT_FLOAT_EQ(Sigmoid::call(-5.0), 0.00669285092428485555936);
	_EXPECT_FLOAT_EQ(Sigmoid::derivative(-5.0), 0.006648056670790154);
}

TEST(Activations, Unzero_and_ClampExtremes) {
	auto uz1 = Unzero<Identity>::call(0.0f);
	EXPECT_GT(uz1, 0.0f);
	EXPECT_LT(uz1, 1.0f);
	auto uz2 = Unzero<Identity>::call(-1.0f);
	_EXPECT_FLOAT_EQ(uz2, -1.0f);
	_EXPECT_FLOAT_EQ(Unzero<Identity>::derivative(-1.0f), 1.0f);

	EXPECT_LT(ClampExtremes<Identity>::call(std::numeric_limits<float>::infinity()),
			std::numeric_limits<float>::infinity());
	EXPECT_GT(ClampExtremes<Identity>::call(-std::numeric_limits<float>::infinity()),
			-std::numeric_limits<float>::infinity());
	_EXPECT_FLOAT_EQ(ClampExtremes<Identity>::derivative(-1.0f), 1.0f);
}
}

// ===== TEST LAYERS =====
namespace seimei::nnet {
TEST(Dense_Layer, Creation) {
	layers::DenseBlueprint blueprint{"tanh/uz"};
	std::string original_name {"Test Layer"};
	auto shaped_bp = blueprint.makeShaped(original_name, 100, 100);
	std::string name {};
	size_t input_size {0};
	size_t output_size {0};
	shaped_bp->getShape(name, input_size, output_size);
	EXPECT_EQ(name, original_name);
	EXPECT_EQ(input_size, 100);
	EXPECT_EQ(output_size, 100);
	LayerMemoryRequirements memreqs {};
	shaped_bp->getMemoryRequirements(memreqs);
	EXPECT_EQ(memreqs.numTempStateForward, 0);
	EXPECT_EQ(memreqs.numTempStateBackward, 0);
	EXPECT_EQ(memreqs.szPersistent, 10000 * sizeof(FLOAT));

	// Test the rest only for throws
	auto pool = shaped_bp->createPool("Test Network", nullptr);
	std::array<uint64_t, 4> rgen {1, 2, 3, 4};
	pool->initializeWeights(rgen);

	std::array<FLOAT, 100 + 100 + 10000> values {0};
	pool->createInstance(true, values.data(), values.data() + 100, values.data() + 200);
}
}

// ===== TEST NNET CONSTRUCTION =====
// This tests the implementation, but I feel unsafe without testing it
namespace seimei::nnet {
class NNetTestInternal : public ::testing::Test {
protected:
	static void verifySingleLayer(NetworkBlueprint& nbp, bool bias) {
		EXPECT_EQ(nbp.getInputSize(), 100);
		EXPECT_EQ(nbp.getOutputSize(), 200);

		auto& compilationTrainable = *nbp.compilationTrainable;
		ASSERT_EQ(compilationTrainable.size(), 1);
		auto& cnode = compilationTrainable.at(0);
		EXPECT_EQ(cnode.inputAt_where, NetworkBlueprint::StorageType::NETIN);
		EXPECT_EQ(cnode.inputAt_start, (bias ? 0 : 1));
		EXPECT_EQ(cnode.biasAt, 0);
		EXPECT_EQ(cnode.outputAt_where, NetworkBlueprint::StorageType::NETOUT);
		EXPECT_EQ(cnode.outputAt_start, 0);
		auto lbp_out = dynamic_cast<layers::DenseBlueprint*>(cnode.blueprint.get());
		EXPECT_TRUE(lbp_out);

		EXPECT_EQ(nbp.realInputSizeTrainable, 101);
		EXPECT_EQ(nbp.realInputSizeFwdOnly, 101);
		EXPECT_EQ(nbp.tempSizeTrainable, 0);
		EXPECT_EQ(nbp.tempSizeFwdOnly, 0);
	}

	static void test_SingleLayer(bool bias) {
		NetworkBlueprint nbp {"Test Network", 100};
		layers::DenseBlueprint lbp {"identity"};
		auto next_id = nbp.addLayer(lbp, "Test Layer", 0, bias, 200);
		ASSERT_EQ(next_id, NetworkBlueprint::FIRST_FREE_OUTPUT);
		ASSERT_ANY_THROW(nbp.getOutputSize());
		nbp.setNetworkOutput(next_id);

		verifySingleLayer(nbp, bias);
		auto serialization = nbp.serializeAsString();
		std::cerr << "SERIALIZATION: " << serialization << std::endl;
		auto ptr_nbp2 = NetworkBlueprint::deserialize(serialization);
		verifySingleLayer(*ptr_nbp2, bias);
	}

	static void verifyDoubleLayer(NetworkBlueprint& nbp) {
		EXPECT_EQ(nbp.getInputSize(), 100);
		EXPECT_EQ(nbp.getOutputSize(), 50);

		auto& compilationTrainable = *nbp.compilationTrainable;
		ASSERT_EQ(compilationTrainable.size(), 2);

		auto& cnode = compilationTrainable.at(0);
		EXPECT_EQ(cnode.inputAt_where, NetworkBlueprint::StorageType::NETIN);
		EXPECT_EQ(cnode.inputAt_start, 1);
		EXPECT_EQ(cnode.biasAt, 0);
		EXPECT_EQ(cnode.outputAt_where, NetworkBlueprint::StorageType::TEMP);
		EXPECT_EQ(cnode.outputAt_start, 1);
		auto lbp_out = dynamic_cast<layers::DenseBlueprint*>(cnode.blueprint.get());
		EXPECT_TRUE(lbp_out);

		auto& cnode2 = compilationTrainable.at(1);
		EXPECT_EQ(cnode2.inputAt_where, NetworkBlueprint::StorageType::TEMP);
		EXPECT_EQ(cnode2.inputAt_start, 0);
		EXPECT_EQ(cnode2.biasAt, 0);
		EXPECT_EQ(cnode2.outputAt_where, NetworkBlueprint::StorageType::NETOUT);
		EXPECT_EQ(cnode2.outputAt_start, 0);
		lbp_out = dynamic_cast<layers::DenseBlueprint*>(cnode.blueprint.get());
		EXPECT_TRUE(lbp_out);

		EXPECT_EQ(nbp.realInputSizeTrainable, 101);
		EXPECT_EQ(nbp.realInputSizeFwdOnly, 101);
		EXPECT_EQ(nbp.tempSizeTrainable, 201);
		EXPECT_EQ(nbp.tempSizeFwdOnly, 201);
	}

	static void test_DoubleLayer() {
		NetworkBlueprint nbp {"Test Network", 100};
		layers::DenseBlueprint lbp {"identity"};
		auto next_id = nbp.addLayer(lbp, "Test Layer 1", 0, false, 200);
		ASSERT_EQ(next_id, NetworkBlueprint::FIRST_FREE_OUTPUT);
		ASSERT_ANY_THROW(nbp.addLayer(lbp, "Test Layer 2", 0, true, 50));
		ASSERT_ANY_THROW(nbp.addLayer(lbp, "Test Layer 2", 42, true, 50));
		next_id = nbp.addLayer(lbp, "Test Layer 2", next_id, true, 50);
		ASSERT_EQ(next_id, NetworkBlueprint::FIRST_FREE_OUTPUT + 1);
		ASSERT_ANY_THROW(nbp.setNetworkOutput(NetworkBlueprint::FIRST_FREE_OUTPUT));
		ASSERT_ANY_THROW(nbp.getOutputSize());
		nbp.setNetworkOutput(next_id);

		verifyDoubleLayer(nbp);
		auto serialization = nbp.serializeAsString();
		std::cerr << "SERIALIZATION: " << serialization << std::endl;
		auto ptr_nbp2 = NetworkBlueprint::deserialize(serialization);
		verifyDoubleLayer(*ptr_nbp2);
	}

	static void test_Complex() {
		/*
		 *               input (30)
		 *                   |
		 *                splitter
		 *                   |
		 *       10      and   12  and   8
		 *        |            |         |
		 *     copier       discard      |
		 *    |     |                    |
		 *   10   layer                  |
		 *    |     |                    |
		 *    |    50                    |
		 *    |     |                    |
		 *   <         joiner             >
		 *               |
		 *             layer
		 *               |
		 *           output (1)
		 *
		 * Temporary memory: 11 for the copy of layer input + 69 for the joiner = 80
		 */
		// FIXME: NetworkBlueprint will actually break if any .add* method throws anything
		// (excepting throws due to input ID being invalid or already bound)
		// This is, however, unimportant for typical usage
		NetworkBlueprint nbp {"Test Network", 30};
		// The splitter
		//std::vector<size_t> bad_splitter_sizes_1 {10, 5, 4};
		//std::vector<size_t> bad_splitter_sizes_2 {15, 30, 45};
		std::vector<size_t> splitter_sizes {10, 12, 8};
		std::vector<size_t> splitter_outputs(3);
		//ASSERT_ANY_THROW(nbp.addSplitter(0, bad_splitter_sizes_1.begin(), bad_splitter_sizes_1.end(), splitter_outputs.begin()));
		//ASSERT_ANY_THROW(nbp.addSplitter(0, bad_splitter_sizes_2.begin(), bad_splitter_sizes_2.end(), splitter_outputs.begin()));
		nbp.addSplitter(0, splitter_sizes.begin(), splitter_sizes.end(), splitter_outputs.begin());
		// Copier
		std::vector<size_t> copier_outputs(2);
		//ASSERT_ANY_THROW(nbp.addCopier(0, copier_outputs.begin(), copier_outputs.end()));
		nbp.addCopier(splitter_outputs.at(0), copier_outputs.begin(), copier_outputs.end());
		// Discarder
		//ASSERT_ANY_THROW(nbp.addDiscarder(splitter_outputs.at(0)));
		nbp.addDiscarder(splitter_outputs.at(1));
		// Hidden layer
		layers::DenseBlueprint hlbp {"sigmoid"};
		auto hl_out = nbp.addLayer(hlbp, "Hidden Layer", copier_outputs.at(1), true, 50);
		// Joiner
		//std::vector<size_t> bad_joiner_inputs {copier_outputs.at(0), copier_outputs.at(1), splitter_outputs.at(2)};
		std::vector<size_t> joiner_inputs {copier_outputs.at(0), hl_out, splitter_outputs.at(2)};
		//ASSERT_ANY_THROW(nbp.addJoiner(bad_joiner_inputs.begin(), bad_joiner_inputs.end()));
		auto joiner_out = nbp.addJoiner(joiner_inputs.begin(), joiner_inputs.end());
		// Output layer
		layers::DenseBlueprint olbp {"tanh/uz"};
		auto ol_out = nbp.addLayer(olbp, "Output Layer", joiner_out, true, 1);
		nbp.setNetworkOutput(ol_out);

		auto serialization = nbp.serializeAsString();
		std::cerr << "SERIALIZATION: " << serialization << std::endl;

		EXPECT_EQ(nbp.getInputSize(), 30);
		EXPECT_EQ(nbp.getOutputSize(), 1);

		EXPECT_EQ(nbp.realInputSizeTrainable, 33);
		EXPECT_EQ(nbp.tempSizeTrainable, 80);

		// TODO: verify the actual compilation

		auto nbp2 = NetworkBlueprint::deserialize(serialization);

		EXPECT_EQ(nbp2->getInputSize(), 30);
		EXPECT_EQ(nbp2->getOutputSize(), 1);

		EXPECT_EQ(nbp2->realInputSizeTrainable, 33);
		EXPECT_EQ(nbp2->tempSizeTrainable, 80);
	}
};

TEST_F(NNetTestInternal, Single_Layer_Bias) {
	test_SingleLayer(true);
}

TEST_F(NNetTestInternal, Single_Layer_NoBias) {
	test_SingleLayer(false);
}

TEST_F(NNetTestInternal, Double_Layer) {
	test_DoubleLayer();
}

TEST_F(NNetTestInternal, Complex_Network) {
	test_Complex();
}
}
