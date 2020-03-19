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
 * This file contains a framework for the actual neural network
 * implementations.
 *
 * To create a new neural network:
 *
 * 1. Construct a NetworkBlueprint.
 * 2. Construct instances of subclasses of LayerBlueprint.
 * 3. Add layers to the network using NetworkBlueprint.addLayer.
 * 4. Call NetworkBlueprint.setNetworkOutput.
 * 5. Call NetworkBlueprint.createPool to get a NetworkPool.
 * 6. (If necessary) Call NetworkPool.initializeWeights.
 * 7. Call NetworkPool.createInstance once or more times to create usable
 * instances. Instances share weights but have separate internal states.
 * 8. Run the instances by calling NetworkInstance.forward, train them by
 *     calling NetworkInstance.forward and then NetworkInstance.backward.
 *     Instances can be used in parallel.
 * 9. (if training) Call NetworkPool.updateWeights after each batch.
 *
 * To serialize a neural network, call NetworkBlueprint.serialize (to serialize
 * without weights) or NetworkPool.serialize (which will also serialize the
 * blueprint if necessary).
 *
 * To deserialize a neural network, call NetworkBlueprint::deserialize and
 * then createPool with the handle of the group with the weights.
 *
 * Programs using this header must link with nnet/framework.cpp and the
 * HDF5 C++ library.
 */

#ifndef SEIMEI_NNET_FRAMEWORK_HPP_
#define SEIMEI_NNET_FRAMEWORK_HPP_

#include "common.hpp"

#include <H5Cpp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <list>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace seimei::nnet {
using std::array;
using std::set;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

/**
 * Check if the string is a suitable network or layer name.
 *
 * @return true if the string is non-empty, consists of ASCII [a-zA-Z0-9-_]
 *     and spaces, and neither starts nor ends with a space; false otherwise.
 */
bool checkIdentifier(const string& s);

/**
 * Like checkIdentifier, but throw std::invalid_argument if not suitable.
 */
void assertIdentifier(const string& s);

/**
 * Throw std::runtime_error if a * b * sizeof(FLOAT) is not representable as
 * size_t or a * b * 2 is not representable as a blasint.
 */
void assertSZMUL(size_t a, size_t b);

/**
 * Like a BLAS AXPY but make sure the output is a finite value.
 *
 * @param    alpha     A scalar
 * @param    X         A vector of `size` elements
 * @param    size      Size of the vectors
 * @param    result    Another vector of `size` elements
 *
 * When the call returns:
 * * result = alpha * X + result; every element of result will be finite.
 */
inline void clamp_axpy(FLOAT alpha, const FLOAT* X, size_t size, FLOAT* result) {
	std::transform(X, X + size, result, result, [alpha](const FLOAT& a, const FLOAT& b) -> FLOAT {
		return std::clamp(alpha * a + b, std::numeric_limits<FLOAT>::lowest(), std::numeric_limits<FLOAT>::max());
	});
}

/**
 * If the argument is NaN, return it. Otherwise return epsilon multiplied
 * by `factor` and multiply `factor` by -1.
 */
inline FLOAT antinan(FLOAT x, FLOAT& factor) {
	if(std::isnan(x)) {
		x = std::numeric_limits<FLOAT>::epsilon() * factor;
		factor *= -1;
	}
	return x;
}

namespace {
static const FLOAT DEFAULT_ANTINAN_FACTOR {-8};
}

/// Description of a layer's memory requirements (not counting input and output)
struct LayerMemoryRequirements {
	/**
	 * How many FLOATs the layer needs as temporary space during forward calculations.
	 *
	 * Weights and internal state that persists between forward runs is not to be
	 * counted here.
	 */
	size_t numTempStateForward {0};

	/**
	 * How many FLOATs the layer needs as temporary state during
	 * backward calculations.
	 */
	size_t numTempStateBackward {0};

	/**
	 * Size of weights and persistent data in chars (estimated,
	 * for informational purposes only)
	 */
	size_t szPersistent {0};

	/**
	 * Size of delta storage during training in chars (estimated,
	 * for informational purposes only)
	 */
	size_t szDeltas {0};

	/**
	 * Size of internal state that persists between runs in chars
	 * (estimated, not including weights). The exact number is for
	 * informational purposes only, but if it's 0, it will be assumed
	 * the instance doesn't have to reset internal state between runs).
	 */
	size_t szInternalState {0};
};

/// This is for internal use in NetworkBlueprint, NetworkPool and NetworkInstance
enum class NodeType {
	LAYER, COPIER, SPLITTER, JOINER,
};

class NetworkPool;
class NetworkInstance;
class LayerBlueprint;
class LayerPool;
class LayerInstance;

/**
 * A blueprint for a neural network. The blueprint represents only
 * the structure; once the structure is completed, call createNetworkPool
 * to get a network with weights.
 */
class NetworkBlueprint {
	friend NetworkPool;	// Will need to access private types
#ifdef SEIMEI_TESTER_NNET_FRIEND
	friend class SEIMEI_TESTER_NNET_FRIEND;
#endif
public:
	/**
	 * Load a NetworkBlueprint with the given name from the given HDF5 group.
	 * The group will usually be a HDF5 file.
	 */
	static unique_ptr<NetworkBlueprint> deserialize(const string& name, H5::Group& from);
	/// Deserialize a JSON description of the network, provided as a string.
	static unique_ptr<NetworkBlueprint> deserialize(const string& json);
	/// Deserialize a JSON description of the network, provided as a JSON object.
	static unique_ptr<NetworkBlueprint> deserialize(const nlohmann::json& json);

	/**
	 * Create a blueprint. Note that you will need a unique_ptr to it
	 * to create a NetworkPool.
	 *
	 * @param    name          Name of the network. Must pass \ref checkIdentifier.
	 * @param    input_size    Size of input for the network
	 */
	NetworkBlueprint(const string& name, size_t input_size);
	NetworkBlueprint(const NetworkBlueprint& other) = delete;
	NetworkBlueprint(NetworkBlueprint&& other) = default;
	NetworkBlueprint& operator=(const NetworkBlueprint& other) = delete;
	NetworkBlueprint& operator=(NetworkBlueprint&& other) = default;
	~NetworkBlueprint() = default;

	/**
	 * @return input size of the network
	 */
	size_t getInputSize() const;

	/**
	 * @return output size of the network
	 * @throws std::logic_error if setOutput has not been called yet
	 */
	size_t getOutputSize() const;

	/**
	 * Add a layer to the neural network, connecting its input to the
	 * specified output.
	 *
	 * Network output must NOT yet be set.
	 *
	 * @param    layer          The layer to add
	 * @param    name           Name to assign to the layer. Must pass
	 *     \ref checkIdentifier and be unique in the network.
	 * @param    input_id       ID of the output to use as input for this layer.
	 *     If 0, then the input of the network will be used. The output must
	 *     not already be used as input for any layer.
	 * @param    bias           If true, add a bias neuron to the input.
	 * @param    output_size    Size of the layer's output. Must not be 0.
	 *
	 * @return ID assigned to the layer's output to use as input for another layer.
	 */
	size_t addLayer(const LayerBlueprint& layer, const std::string& name, size_t input_id, bool bias,
			size_t output_size);
	/**
	 * Add a "copier" that takes a single input and copies it, so multiple
	 * layers can use the same input.
	 *
	 * Network output must NOT yet be set.
	 *
	 * @param    output_ids_start   Where to start copying the new output IDs.
	 * @param    output_ids_end     Iterator past the end for output IDs.
	 *     The number of outputs will be output_ids_end - output_ids_start.
	 *     It must not be >= 2.
	 */
	void addCopier(size_t input_id, vector<size_t>::iterator output_ids_start, vector<size_t>::iterator output_ids_end);

	/**
	 * Add a "splitter" that takes a single input and splits it into multiple
	 * outputs to use as inputs for multiple layers.
	 *
	 * Network output must NOT have been set.
	 *
	 * @param    sizes_start         Start of the list of sizes. The sum of
	 *     all sizes must equal the size of the input. No size must be 0.
	 *     The total number of outputs must be >= 2.
	 * @param    sizes_end           Iterator past the end for sizes.
	 * @param    output_ids_start    Where to start copying the output IDs.
	 *     There must be enough space for all sizes.
	 */
	void addSplitter(size_t input_id, vector<size_t>::iterator sizes_start, vector<size_t>::iterator sizes_end,
			vector<size_t>::iterator output_ids_start);

	/**
	 * Add a "joiner" that takes multiple inputs and joins them together
	 * as one output.
	 *
	 * Network output must NOT yet be set.
	 *
	 * @param    input_ids_start    Iterator to the start of the list of inputs.
	 * @param    input_ids_end      Iterator past the end of the list.
	 *
	 * @return ID of the combined output.
	 */
	size_t addJoiner(vector<size_t>::iterator input_ids_start, vector<size_t>::iterator input_ids_end);

	/**
	 * Have a layer output discarded. Probably only a good idea if discarding
	 * part of a splitter's output.
	 *
	 * Network output must NOT hyet be set.
	 *
	 * @param    input_id    ID of the input to discard
	 */
	void addDiscarder(size_t input_id);

	/**
	 * Set the output of a particular layer, splitter or joiner (or copier,
	 * if you want to for some reason) as the output for the entire network.
	 *
	 * This function must be called when the network input has already been
	 * connected to something and every output of every layer (except the one
	 * being set as network output) has been connected to some input.
	 *
	 * The network will then be prepared for creation of pools. No more layers
	 * or helper nodes can be added afterwards.
	 */
	void setNetworkOutput(size_t input_id);

	/**
	 * Create a NetworkPool from this blueprint. The network must have at
	 * least one layer, every output must be connected to a layer, to a
	 * copier/splitter/joiner or set as the network output.
	 *
	 * The pool will be independent from the blueprint. Some caching
	 * may need to be done in the blueprint, however, so it's not a const
	 * method.
	 *
	 * The network output must already be set.
	 *
	 * @param    weights_from   Load weights from this HDF5 group.
	 *     If nullptr, initialize weights with zeros. The group will
	 *     usually be an H5::H5File.
	 */
	shared_ptr<NetworkPool> createPool(H5::Group* weights_from);

	/**
	 * Calculate memory requirements for the network.
	 * It may take a while to calculate. The results will be cached.
	 * The network must be in a state where createPool could be called.
	 *
	 * All results are in bytes.
	 *
	 * The network output must already be set.
	 *
	 * @param[out]    sz_weights         Weights and persistent data
	 * @param[out]    sz_pi_training     Per-instance if trainable
	 * @param[out]    sz_pi_fwdonly      Per-instance if non-trainable
	 *     (supporting only forward runs)
	 */
	void calculateMemoryRequirements(size_t& sz_weights, size_t& sz_pi_training, size_t& sz_pi_fwdonly);

	/**
	 * Serialize the network as a JSON object. The output must already be set.
	 */
	nlohmann::json serialize();

	/**
	 * Serialize the network as a JSON object, then convert it into a string.
	 * The network must have its output set.
	 */
	string serializeAsString();

private:
	inline static const size_t NO_OUTPUT {0};
	inline static const size_t NETWORK_IN_OUT {1};
	inline static const size_t DISCARD_OUTPUT {2};
	inline static const size_t FIRST_FREE_OUTPUT {3};

	inline static const size_t MAX_NODES {0xFFFF - FIRST_FREE_OUTPUT};

	struct LayerNode {
		bool bias {false};
		size_t inputID {0};
		size_t outputID {0};
		unique_ptr<LayerBlueprint> blueprint {};
		std::string name {};
		LayerMemoryRequirements memreqs {};
	};

	struct CopierSplitterNode {
		size_t inputID {0};
		vector<std::pair<size_t, size_t>> targets {};
	};

	struct JoinerNode {
		vector<size_t> inputIDs {};
		size_t outputID {0};
	};

	struct Node {
		NodeType type;
		size_t numInput {0}; // NOT including bias
		size_t numOutput {0};
		union {
			LayerNode ln;
			CopierSplitterNode csn;
			JoinerNode jn;
		};

		Node(NodeType type);
		Node(const Node&);
		Node(Node&&);
		Node& operator=(const Node& other);
		Node& operator=(Node&& other);
		~Node();
	};

	// OUTPUT1 and OUTPUT2 are never used in NetworkPool, only by NetworkBlueprint
	enum class StorageType {
		UNUSED, TEMP, NETIN, NETOUT, OUTPUT1, OUTPUT2
	};

	struct NodeAllocation {
		StorageType where;
		size_t index;
		size_t remainingJoinerInputsP1 {0};	// Set to 1 if all targets are available but the joiner isn't compiled yet
		NodeAllocation(StorageType where, size_t index) :
				where(where), index(index) {
		}
		NodeAllocation() :
				NodeAllocation(StorageType::UNUSED, 0) {
		}
	};

	enum class BackwardBehavior {
		CALL_OR_COPY, ADD_DERIVS
	};

	struct CompiledNode {
		BackwardBehavior BB {BackwardBehavior::CALL_OR_COPY};
		StorageType inputAt_where {StorageType::UNUSED};
		StorageType outputAt_where {StorageType::UNUSED};
		std::vector<size_t> outputSlicing {};
		unique_ptr<LayerBlueprint> blueprint {nullptr};
		shared_ptr<LayerPool> layer {nullptr};
		size_t biasAt {0};
		size_t inputAt_start {0};
		size_t outputAt_start {0};
		size_t inputSize {0};	// Not used if layer is not nullptr

		CompiledNode() = default;
		CompiledNode& operator=(const CompiledNode& other);
		CompiledNode& operator=(CompiledNode&& other) = default;
		CompiledNode(const CompiledNode& other);
		CompiledNode(CompiledNode&& other) = default;
	};

	enum class SerializedIdentifierType {
		NETWORK_IN_OUT,
		HELPER_NODE,
		LAYER
	};

	// Translate a node ID into index, assuming the ID is valid
	static size_t ID2Index(size_t ID);

	/**
	 * Deserialization helper function.
	 *
	 * If the identifier is invalid, throw an std::runtime_error.
	 * If the identifier is valid, remove the ":<number>" part from
	 * it and save the number in `slot`. Otherwise write 0 there. Throw
	 * and std::runtime_error if the slot is actually 0.
	 *
	 * @return the type of the identifier
	 */
	static SerializedIdentifierType parseSerializedIdentifier(string& which, size_t& slot);

	string networkName;
	size_t inputSize;
	size_t outputSize {0};
	size_t networkInputID {NO_OUTPUT};	// ID of the node that uses network input
	size_t networkOutputID {NO_OUTPUT};	// ID of the node that produces network output
	std::unordered_map<string, size_t> name2ID {};
	std::vector<Node> nodes {};
	unique_ptr<vector<CompiledNode>> compilationTrainable {};
	unique_ptr<vector<CompiledNode>> compilationFwdOnly {};
	LayerMemoryRequirements totalMemoryRequirements {};
	size_t realInputSizeTrainable {0};
	size_t realInputSizeFwdOnly {0};
	size_t tempSizeTrainable {0};
	size_t tempSizeFwdOnly {0};

	// Throw an std::logic_error if the network output is or is not set.
	void assertNetworkOutput(bool is_set) const;

	/**
	 * Serialization helper: return the serialized name of an input,
	 * output or a node.
	 */
	string serializeIONName(size_t ion_id) const;

	/**
	 * Serialize discarded outputs for a copier or a joiner.
	 */
	std::vector<size_t> serializeDiscardedOutputs(const vector<std::pair<size_t, size_t>>& targets) const;

	/**
	 * Return the (index + FIRST_FREE_OUTPUT) of the copier/splitter node
	 * with input_id. If it isn't a copier/splitter node, return 0. Throw
	 * std::runtime_error if the ID is invalid.
	 *
	 * If must_be_free is true, check if the output target is not bound,
	 * and throw std::runtime_error if it is. No checks will have been
	 * performed if the call returns 0.
	 *
	 * If the call doesn't return 0, target_index will be set to
	 * the 0-based index of the output target within the node
	 * according to input_id.
	 */
	size_t getCSOutput(size_t input_id, size_t& target_index, bool must_be_free = false);

	/**
	 * Return the next input ID. Throw an std::runtime_error if
	 * the maximum number of nodes has been reached.
	 */
	size_t getNextOutputID() const;

	/**
	 * Attach an output with the given ID to a node. Throw an
	 * exception (std::logic_error or std::runtime_error) if the
	 * node doesn't exist.
	 *
	 * Throw std::logic_error if the output is already set.
	 *
	 * Return the output size.
	 */
	size_t setOutput(size_t input_id, size_t new_output_id);

	/**
	 * Compile a trainable version of the network.
	 * Checks should have been done by the caller.
	 */
	void compileTrainable();

	/**
	 * Compile the forward-only version. A trainable version
	 * should be compiled beforehand.
	 */
	void compileForwardOnly();

	/**
	 * Return node index and calculate allocation index
	 */
	static size_t getAllocationIndex(size_t input_id, size_t& idx_allocation);

	/**
	 * Find the allocation, assuming it exists
	 */
	static NodeAllocation& findAllocation(vector<vector<NodeAllocation>>& allocations, size_t input_id);

	/**
	 * Allocate memory for a joiner If the joiner has a splitter as the output,
	 * allocate memory for the splitter in the output itself.
	 *
	 * Return immediately if the space for the joiner has already been allocated.
	 */
	void startCompilingJoiner(size_t& storage_size, vector<vector<NodeAllocation>>& allocations, size_t idx_joiner);

	/**
	 * Call startCompilingJoiner and then, whether the joiner had previously been
	 * allocated or not, check whether any inputs have become available and
	 * insert copying CompileNodes.
	 *
	 * @return if the joiner is ready for compilation but not compiled yet
	 */
	bool updateJoiner(size_t& storage_size, vector<CompiledNode> compilation,
			vector<vector<NodeAllocation>>& allocations, size_t idx_joiner);

	/**
	 * Attach an input to an already allocated joiner after the input becomes
	 * available. If the joiner has become complete, push it onto the stack.
	 */
	void attachJoinerInput(vector<vector<NodeAllocation>>& allocations, std::list<size_t>& stack,
			size_t joiner_input_id, CompiledNode& cnode);

	/**
	 * Create an output splitting scheme for a CompiledNode so as to implement a
	 * splitter. Push the outputs onto the stack.
	 *
	 * @return the number of splits, which will equal the number of bias inputs
	 * to add
	 */
	size_t compileSplitOutput(vector<vector<NodeAllocation>>& allocations, std::list<size_t>& stack,
			size_t idx_splitter, vector<size_t>& output_slicing, StorageType startAt_where, size_t startAt_index);

	/**
	 * Compile a copier for a trainable network. Automatically split every output
	 * that is connected to a splitter and add the targets to the stack. If an output
	 * is connected to a joiner, allocate it and copy the output directly into it.
	 * Add all other targets to the stack.
	 */
	void compileCopierTrainable(size_t& storage_size, vector<CompiledNode> compilation,
			vector<vector<NodeAllocation>>& allocations, std::list<size_t>& stack, size_t idx_copier,
			StorageType inputAt_where, size_t inputAt_index);
};

/**
 * A neural network with all persistent parameters. It's a factory for
 * instances that do the actual work.
 *
 * WARNING ON MULTITHREADING:
 * * Non-const methods cannot be called in parallel. Further, no methods
 *      of NetworkInstances, not even their const methods can be running
 *      in parallel with non-const methods of NetworkPool.
 * * const methods can be called in parallel with each other but not
 *      with non-const methods.
 * * If a LayerPool is extracted with getLayerPool, the same applies
 *      to its const and non-const methods.
 */
class NetworkPool: std::enable_shared_from_this<NetworkPool> {
	// NetworkBlueprint will have to call the private constructor
	friend NetworkBlueprint;
#ifdef SEIMEI_TESTER_NNET_FRIEND
	friend class SEIMEI_TESTER_NNET_FRIEND;
#endif
public:
	NetworkPool(NetworkPool&) = delete;
	NetworkPool(NetworkPool&&) = delete;
	NetworkPool& operator=(NetworkPool&) = delete;
	NetworkPool& operator=(NetworkPool&&) = delete;

	/**
	 * Discard current weights and reinitialize them with the provided
	 * random seed (the seed will be changed by the random number generator).
	 * The initialization function is to be set for each layer per its own
	 * interface (if it can be changed for a particular layer).
	 */
	void initializeWeights(array<uint64_t, 4>& seed);

	/**
	 * Give a pointer to a LayerPool for a layer in the network. Note that
	 * this won't be a copy, but what the network actually uses.
	 */
	shared_ptr<LayerPool> getLayerPool(const std::string& name) const;

	/**
	 * Create an instance of this network. The instance will contain pointers
	 * to the network itself and use is weights.
	 *
	 * This method behaves like a const method for the purposes of concurrency.
	 */
	unique_ptr<NetworkInstance> createInstance(bool trainable);

private:
	const string networkName, blueprintSerialization;
	const size_t inputSize, outputSize, realInputSizeTrainable, realInputSizeFwdOnly;
	const size_t numTempTrainable, numTempFwdOnly, numScratchTrainable, numScratchFwdOnly;
	std::vector<NetworkBlueprint::CompiledNode> nodesTrainable, nodesFwdOnly;
	std::unordered_map<string, shared_ptr<LayerPool>> name2pool;

	// num_temp_* = number of FLOATs of temporary memory (storage, outputs)
	// num_scratch_* = number of FLOATs of scratch memory
	// WARNING: nodes_trainable and nodes_fwdonly must use the same layer pools
	NetworkPool(const string& network_name, H5::Group* weights_from, const string& blueprint_serialization, size_t input_size, size_t output_size,
			size_t real_input_size_trainble, size_t real_input_size_fwdonly,
			size_t num_temp_trainable, size_t num_temp_fwd_only,
			size_t num_scratch_trainable, size_t num_scratch_fwd_only,
			const vector<NetworkBlueprint::CompiledNode>& nodes_trainable,
			const vector<NetworkBlueprint::CompiledNode>& nodes_fwdonly);
};

/**
 * A neural network instance. It is produced by, and bound to, a NetworkPool.
 *
 * No two methods of an instance can run concurrently. However, multiple
 * instances bound to the same NetworkPool _can_ run in parallel, provided
 * that:
 *
 * 1. No non-const method (with noted exceptions) of the NetworkPool is running
 * concurrently.
 * 2. You follow the guidelines for the specific method of NetworkInstance.
 */
class NetworkInstance {
	// NetworkPool will access private datatypes and call the private constructor
	friend NetworkPool;
#ifdef SEIMEI_TESTER_NNET_FRIEND
	friend class SEIMEI_TESTER_NNET_FRIEND;
#endif
public:
	NetworkInstance(NetworkInstance&) = delete;
	NetworkInstance(NetworkInstance&&) = delete;
	NetworkInstance& operator=(NetworkInstance&) = delete;
	NetworkInstance& operator=(NetworkInstance&&) = delete;
	~NetworkInstance() = default;

	/**
	 * Get the input size and location. The pointer will be valid
	 * for as long as the NetworkInstance exists. The input buffer
	 * is where users must store the input for the network.
	 *
	 * Can run concurrently with anything.
	 *
	 * @param[out]    ptr    Where to save the pointer to the input
	 *     buffer. If nullptr, the pointer will not be copied.
	 */
	size_t getInput(FLOAT** ptr);

	/**
	 * Get the output size and location. The pointer will be valid
	 * for as long as the NetworkInstance exists. The network will
	 * store its output in this buffer.
	 *
	 * Can run concurrently with anything.
	 *
	 * @param[out]    ptr    Where to save the pointer to the
	 *     output buffer. If nullptr, the pointer will not be copied.
	 */
	size_t getOutput(FLOAT** ptr);

	/**
	 * Return true if the instance is trainable, false otherwise.
	 *
	 * Can run concurrently with anything.
	 */
	bool isTrainable() const;

	/**
	 * Run the network forward. The caller must have copied the
	 * input into the input buffer (see \ref getInput). After the call
	 * returns, the output will be in the output buffer (see \ref getOutput).
	 * The input buffer may be altered.
	 *
	 * The internal state of layers that have one will change. A trainable
	 * network will also prepare for a backward run on the same data.
	 *
	 * Can run concurrently with .get*- and .is*-methods, .forward and
	 * .backward of other instances.
	 */
	void forward();

	/**
	 * Run the network backward and update deltas internally. The network
	 * instance must be trainable.
	 *
	 * The network must have been run forward at least once since creation
	 * or last reset; the input and output buffers must contain the exact
	 * same data they did after the last forward run. You may not copy and
	 * restore the buffers, you have to run the network forward and then
	 * backward on the same data. You also must not call resetState
	 * in-between.
	 *
	 * When this function returns, the input buffer will contain
	 * df/dIV f(IV) where f is the network and IV the input vector.
	 *
	 * Can run concurrently with .get*- and .is*-methods, forward and
	 * backward of other instances.
	 *
	 * @param    expected    Expected output values (the array must have
	 *     the same size as the output buffer)
	 */
	void backward(FLOAT lrate, const FLOAT* expected);

	/**
	 * Run the network backward with provided derivatives with respect
	 * to the output. Otherwise like `backward`.
	 */
	void backwardGradient(FLOAT lrate, const FLOAT* derivs);

	/**
	 * Update the weights of the associated pool assuming that this
	 * instance has completed the given part of the total run. The
	 * argument should, mathematically, be between 0.0 and 1.0 inclusive.
	 *
	 * Cannot run concurrently with any method of any other instance associated
	 * with the same pool (except get* and .is*) or any non-const method of the
	 * pool itself.
	 */
	void updateWeights(FLOAT proportion);

	/**
	 * Reset the internal state of recurrent layers.
	 *
	 * Can run concurrently with .get*- and .is*-methods, .forward
	 * and .backward of other instances.
	 */
	void resetState();

private:
	struct Node {
		std::vector<size_t> outputSlicing;
		bool backwardAddDerivs;
		// If layer is nullptr, assume a layer that copies input to output
		unique_ptr<LayerInstance> layer;
		size_t inputSize;
		FLOAT* biasAt, * inputAt, * outputAt;
	};

	bool trainable;
	const size_t inputSize, outputSize;
	vector<Node> nodes;
	vector<FLOAT> input, output, temp, scratch;

	NetworkInstance(bool trainable, size_t input_size,
			size_t output_size,
			vector<FLOAT>&& input,
			vector<FLOAT>&& output, vector<FLOAT>&& temp,
			vector<FLOAT>&& scratch, vector<Node>&& nodes);
};

/**
 * Abstract class for layer blueprints. Construct subclasses for specific
 * layers.
 */
class LayerBlueprint {
public:
	virtual ~LayerBlueprint() = default;

	/**
	 * Clone this layer blueprint.
	 */
	virtual unique_ptr<LayerBlueprint> clone() const = 0;

	/**
	 * Create a blueprint of the same kind, but with the specific
	 * name and sizes. NetworkBlueprint will call this method.
	 *
	 * Subclasses must support this method even in case the "shape"
	 * had already been set.
	 *
	 * @return LayerBlueprint with the new "shape".
	 */
	virtual unique_ptr<LayerBlueprint> makeShaped(const std::string& name, size_t input_size,
			size_t output_size) const = 0;

	/**
	 * Get the shape parameters for the layer. Parameters will be
	 * assigned the corresponding values. If the shape has not been
	 * set, parameters will be unchanged.
	 *
	 * @return true if the shape actually has been set, false if not.
	 */
	virtual bool getShape(std::string& name, size_t& input_size, size_t& output_size) const = 0;

	/**
	 * Calculate memory requirements.
	 *
	 * @param[out]   dest   Structure to write the result into
	 *
	 * @throw std::logic_error if the layer has no shape and can't
	 *     calculate requirements without it.
	 */
	virtual void getMemoryRequirements(LayerMemoryRequirements& dest) const = 0;

	/**
	 * Create a pool for this layer. The pool must be independent from
	 * the blueprint.
	 *
	 * @param    network_name    Name of the entire network. Must pass
	 *     checkIdentifier.
	 * @param    weights_from    Load weights from this group (layer has
	 *     free reign under ``seimei/networks/<network_name>/<FLOAT_TEXT>/layers/<layer name>``).
	 *     ``FLOAT_TEXT`` is a macro defined in common.hpp as a string literal.
	 *     If nullptr, initialize weights with zeros.
	 * @throw std::logic_error if the layer has no shape but needs
	 *     one.
	 */
	virtual shared_ptr<LayerPool> createPool(const std::string& network_name, H5::Group* weights_from) const = 0;

	/**
	 * Serialize the blueprint into the provided JSON object (which will actually
	 * be an object, .is_object() will return true). The object will already
	 * have some fields set by the caller, and they are not to be altered:
	 *
	 * * ``"type":`` ``"layer"``
	 * * ``"input":`` (value must be ignored and not used)
	 * * ``"name"``: name of the layer
	 * * ``"output_size"``: integer value
	 * * ``"discard_output"``: boolean value
	 *
	 * The blueprint must set the key ``"layer_type"`` to its type,
	 * e.g. ``"dense"``. The value must be one of those recognized
	 * by deserializeBlueprint in framework.cpp. The blueprint
	 * must further set any other fields as it needs to represent
	 * itself.
	 */
	virtual void serialize(nlohmann::json& out) = 0;

	/**
	 * Deserialize the blueprint from the provided JSON object.
	 * The shape will be set later.
	 *
	 * The caller is supposed to infer the appropriate subclass
	 * and create an instance of it in a valid state, then call
	 * .deserialize on it.
	 */
	virtual unique_ptr<LayerBlueprint> deserialize(const nlohmann::json& from) = 0;
protected:
	static void assertNonZeroShape(const std::string& layer_name, size_t input_size, size_t output_size) {
		if(!input_size || !output_size) {
			throw std::logic_error(
					string("Trying to create the layer \"") + layer_name + std::string("\" without input and/or output"));
		}
	}
};

/**
 * LayerPool is an abstract base class for objects that contain weights,
 * but not internal state, for a particular layer. It must not use the
 * original blueprint.
 *
 * Not two methods will be called concurrently.
 */
class LayerPool {
public:
	virtual ~LayerPool() = default;

	/**
	 * Get the name of the pool.
	 */
	virtual const string& getName() const = 0;

	/**
	 * Reinitialize the weights of this pool using the given random
	 * generator state. Alter the state accordingly.
	 *
	 * No pool instance will be accessing the weights until the method
	 * returns.
	 */
	virtual void initializeWeights(array<uint64_t, 4>& state) = 0;

	/**
	 * Copy weights from the provided LayerPool. If it's not compatible
	 * with this one, throw std::domain_error.
	 *
	 * No pool instance will be accessing the weights until the method
	 * returns.
	 */
	virtual void copyWeights(const LayerPool& from) = 0;

	/**
	 * Save current weights to the HDF5 group. The layer has
	 * free reign under ``seimei/networks/<network_name>/<FLOAT_TEXT>/layers/<layer name>``.
	 * ``FLOAT_TEXT`` is a macro defined in common.hpp as a string literal.
	 *
	 * No pool instance will be accessing the weights until the method
	 * returns.
	 */
	virtual void saveWeights(H5::Group* to) const = 0;

	/**
	 * Create a corresponding LayerInstance. Pass the pointers to it.
	 *
	 * Other instances may be accessing the pool in parallel.
	 * Multiple creation requests can be running in parallel as well.
	 *
	 * @param    trainable     Whether the instance will be part
	 *     of a trainable network.
	 * @param    input         Where the input is to be read from.
	 *     It will be valid while the instance exists.
	 * @param    output        Where the output is to be written to.
	 *     It will be valid while the instance exists.
	 * @param    temp          Where temporary data can be stored
	 *     until .forward or .backward returns. It may be nullptr
	 *     if the layer won't need temporary data. Otherwise the
	 *     size of the buffer will be numTempStateForward if the
	 *     instance is not trainable, max(numTempStateForward, tempStateBackward)
	 *     if it is.
	 */
	virtual unique_ptr<LayerInstance> createInstance(bool trainable, FLOAT* input, FLOAT* output,
	FLOAT* temp) = 0;

protected:
	LayerPool() = default;
};

/**
 * LayerInstance is an abstract base class for objects that store the internal
 * state of a particular layer, if any, and convert input into output.
 * If trainable, a LayerInstance must store its updates (deltas). In any case,
 * it must store its assigned pointers to input and output buffers and a
 * shared pointer to the associated LayerPool, if it needs one.
 *
 * No two methods will be called concurrently.
 */
class LayerInstance {
public:
	virtual ~LayerInstance() = default;

	/**
	 * Run the instance forwards on the input in the input buffer, update
	 * internal state (if any) and store the output in the output buffer.
	 */
	virtual void forward() = 0;

	/**
	 * Run the instance backwards.
	 *
	 * When this method is called, the input buffer will contain the
	 * input given to this layer during the previous forward run, and
	 * each element of the output buffer will contain the derivative
	 * of the loss function with respect to that output (FIXME: not really).
	 *
	 * This method must store updates for its weights (but NOT the actual
	 * weights) and cause each element of the input buffer to contain
	 * the derivative of the loss function with respect to that input.
	 *
	 * If the layer has internal state, the next .forward call, unless
	 * preceded by .resetState, must work as though no .backward call
	 * occurred in-between.
	 *
	 * This method will never be called if the network instance is not
	 * trainable.
	 *
	 * @param    lrate    The learning rate
	 */
	virtual void backward(FLOAT lrate) = 0;

	/**
	 * Update weights of the associated pool and reset stored updates to 0,
	 * assuming that this instance has done a particular part of the total
	 * run (while other instances have done the rest).
	 */
	virtual void updateWeights(FLOAT proportion) = 0;

	/**
	 * Reset internal state of the layer. Return immediately if there is
	 * no internal state. The default implementation returns immediately.
	 *
	 * This might not be called if the layer has 0 chars of internal state.
	 */
	virtual void resetState() {
		return;
	}

protected:
	LayerInstance() = default;
};
}

#endif
