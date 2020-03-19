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

#include "blas.hpp"
#include "nnet/framework.hpp"
#include "nnet/layers.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace seimei::nnet {
namespace {
unique_ptr<LayerBlueprint> deserializeBlueprint(const nlohmann::json& from) {
	unique_ptr<LayerBlueprint> tmp;
	auto ltype = from.at("layer_type").get<std::string>();
	if(ltype == "activation") {
		tmp = std::make_unique<layers::ActivationBlueprint>();
	}
	else if(ltype == "dense") {
		tmp = std::make_unique<layers::DenseBlueprint>();
	} else {
		throw std::logic_error(string("Unknown layer type: ") + ltype);
	}
	return tmp->deserialize(from);
}

bool checkIdentifierChar(char c) {
	return !((c < '0' && c != ' ' && c != '-') || (c > '9' && c < 'A') || (c > 'Z' && c < 'a' && c != '_') || c > 'z');
}
}

bool checkIdentifier(const string& s) {
	if(s.empty()) {
		return false;
	}
	for(char c : s) {
		if(!checkIdentifierChar(c)) {
			return false;
		}
	}
	return true;
}

void assertIdentifier(const string& s) {
	if(!checkIdentifier(s)) {
		throw std::invalid_argument(string("Invalid identifier: ") + s);
	}
}

void assertSZMUL(size_t a, size_t b) {
	bool toobig {false};
	if constexpr (static_cast<size_t>(std::numeric_limits<blasint>::max()) < std::numeric_limits<size_t>::max()) {
		if(std::numeric_limits<blasint>::max() / 2 / a < b) {
			toobig = true;
		}
	}
	size_t maxmul {std::max(sizeof(FLOAT), static_cast<size_t>(2))};
	if(std::numeric_limits<size_t>::max() / maxmul / a < b) {
		toobig = true;
	}
	if(toobig) {
		throw std::runtime_error(
				string("The dimensions ") + std::to_string(a) + string(" x ") + std::to_string(b)
						+ string(" are too big"));
	}
}

unique_ptr<NetworkBlueprint> NetworkBlueprint::deserialize(const nlohmann::json& json) {
	auto result = std::make_unique<NetworkBlueprint>(json.at("name").get<string>(), json["input_size"].get<size_t>());
	std::unordered_map<string, size_t> node_map {};
	auto nodes = json.at("nodes");
	auto seek_input = [&result, &node_map](string& identifier) -> size_t {
		size_t slot;
		auto identifier_type = NetworkBlueprint::parseSerializedIdentifier(identifier, slot);
		if(identifier_type == SerializedIdentifierType::NETWORK_IN_OUT) {
			if(result->networkInputID != NO_OUTPUT) {
				throw std::runtime_error("Network input referenced more than once");
			}
			return 0;
		} else {
			auto it = node_map.find(identifier);
			if(it == node_map.end()) {
				throw std::runtime_error(string("Node referenced before being defined: ") + identifier);
			}
			auto& node = result->nodes.at(it->second - FIRST_FREE_OUTPUT);
			if(node.type == NodeType::LAYER || node.type == NodeType::JOINER) {
				if(slot) {
					throw std::runtime_error(string("Node \"") + identifier + string("\" referenced with  slot number but has no slots"));
				}
				return it->second;
			} else if(node.type == NodeType::COPIER || node.type == NodeType::SPLITTER) {
				if(!slot) {
					throw std::runtime_error(string("Node \"") + identifier + string("\" cannot be referenced without a slot number"));
				} else if(slot > node.csn.targets.size()) {
					throw std::runtime_error(string("Node \"") + identifier + string("\" has only ") + std::to_string(node.csn.targets.size()) + string(" slots"));
				} else if(node.csn.targets.at(slot - 1).first != NO_OUTPUT) {
					throw std::runtime_error(string("Slot \"") + identifier + string(":") + std::to_string(slot) + string("\" referenced more than once"));
				}
				return (it->second << 16) | (slot - 1);
			} else {
				unreachable("NetworkBlueprint::deserialize::seek_input");
			}
		}
	};

	size_t slot;
	for(auto it = nodes.begin(); it != nodes.end(); ++it) {
		const auto& node_json = *it;
		auto name = node_json.at("name").get<string>();
		auto node_name_type = parseSerializedIdentifier(name, slot);
		if(slot) {
			throw std::runtime_error(string("Node name contains a slot number: ") + node_json["name"].get<string>());
		} else if(node_name_type == SerializedIdentifierType::NETWORK_IN_OUT) {
			throw std::runtime_error("Node name is \"@network\"");
		} else if(node_map.find(name) != node_map.end()) {
			throw std::runtime_error(string("Repeated node name: ") + name);
		}
		auto type = node_json.at("type").get<string>();
		size_t new_id;
		if(type == "copier" || type == "splitter") {
			auto identifier = node_json.at("input").get<string>();
			auto input_id = seek_input(identifier);
			std::vector<size_t> output_ids {};
			if(type == "copier") {
				output_ids.resize(node_json.at("num_outputs").get<size_t>());
				result->addCopier(input_id, output_ids.begin(), output_ids.end());
			} else {
				auto output_sizes = node_json.at("output_sizes").get<std::vector<size_t>>();
				output_ids.resize(output_sizes.size());
				result->addSplitter(input_id, output_sizes.begin(), output_sizes.end(), output_ids.begin());
			}
			// We do not save the output IDs because we know how we produce them
			new_id = output_ids.at(0) >> 16;
			auto discard_outputs = node_json.at("discard_outputs").get<std::vector<size_t>>();
			for(auto dout : discard_outputs) {
				if(dout > output_ids.size()) {
					throw std::runtime_error(string("Node \"") + name + string("\": discarding output that doesn't exist: ") + std::to_string(dout));
				}
				result->addDiscarder(output_ids.at(dout - 1));
			}
		} else if(type == "joiner") {
			auto inputs = node_json.at("inputs").get<vector<string>>();
			vector<size_t> input_ids(inputs.size());
			std::transform(inputs.begin(), inputs.end(), input_ids.begin(), [&seek_input](const string& identifier) -> size_t {
				auto tmp = identifier;
				return seek_input(tmp);
			});
			new_id = result->addJoiner(input_ids.begin(), input_ids.end());
			if(node_json.at("discard_output").get<bool>()) {
				result->addDiscarder(new_id);
			}
		} else if(type == "layer") {
			auto identifier = node_json.at("input").get<string>();
			auto input_id = seek_input(identifier);
			auto output_size = node_json.at("output_size").get<size_t>();
			auto ptr_blueprint = deserializeBlueprint(node_json);
			new_id = result->addLayer(*ptr_blueprint, name, input_id, node_json.at("bias").get<bool>(), output_size);
			if(node_json.at("discard_output").get<bool>()) {
				result->addDiscarder(new_id);
			}
		} else {
			throw std::runtime_error(string("Invalid node type \"") + type + string("\""));
		}
		node_map.emplace(name, new_id);
	}

	auto network_output_identifier = json.at("output").get<string>();
	auto network_output_id = seek_input(network_output_identifier);
	result->setNetworkOutput(network_output_id);
	return result;
}

unique_ptr<NetworkBlueprint> NetworkBlueprint::deserialize(const string& json) {
	nlohmann::json json_object(nlohmann::json::parse(json));
	return deserialize(json_object);
}

size_t NetworkBlueprint::ID2Index(size_t ID) {
	if(ID > 0xFFFF) {
		return (ID >> 16) - FIRST_FREE_OUTPUT;
	} else {
		return ID - FIRST_FREE_OUTPUT;
	}
}

NetworkBlueprint::SerializedIdentifierType NetworkBlueprint::parseSerializedIdentifier(string& which, size_t& slot) {
	if(which.empty()) {
		throw std::runtime_error("Empty name or reference");
	} else if(which == "@network") {
		slot = 0;
		return SerializedIdentifierType::NETWORK_IN_OUT;
	} else {
		SerializedIdentifierType result;
		size_t index {0};
		size_t end {which.size()};
		if(which.at(0) == '#') {
			result = SerializedIdentifierType::HELPER_NODE;
			index = 1;
			if(end < 2) {
				throw std::runtime_error("A node name cannot consist of a single \"#\"");
			}
		} else {
			result = SerializedIdentifierType::LAYER;
		}
		size_t slot_start {0};
		while(index < end) {
			auto c = which.at(index++);
			if(c == ':') {
				if(index == (result == SerializedIdentifierType::LAYER ? 1 : 2)) {
					throw std::runtime_error(string("No name before slot number in \"") + which + string("\""));
				} else if(index == end) {
					throw std::runtime_error(string("Nothing after \":\" in \"") + which + string("\""));
				}
				slot_start = index;
				break;
			} else if(!checkIdentifierChar(c)) {
				throw std::runtime_error(string("Invalid node name or reference: ") + which);
			}
		}
		if(slot_start) {
			while(index < end) {
				auto c = which.at(index++);
				if(c < '0' || c > '9') {
					throw std::runtime_error(string("Invalid node name or reference: ") + which);
				}
			}
			try {
				auto slot_id = std::stoi(which.substr(slot_start));
				if(slot_id < 0) {
					// This shouldn't be possible, but what if...
					throw std::out_of_range("slot_id < 0");
				} else if(slot_id == 0) {
					throw std::runtime_error(string("Slot number is 0, which is not allowed: ") + which);
				}
				slot = static_cast<size_t>(slot_id);
			} catch(std::out_of_range& e) {
				throw std::runtime_error(string("Slot number too big: ") + which);
			}
		} else {
			slot = 0;
		}
		which = which.substr(0, slot_start - 1);
		return result;
	}
}

NetworkBlueprint::NetworkBlueprint(const string& name, size_t input_size) :
		networkName(name), inputSize(input_size) {
	assertIdentifier(name);
}

NetworkBlueprint::Node::Node(NodeType type) :
		type(type) {
	switch(type) {
	case NodeType::LAYER:
		new (&this->ln) LayerNode {};
		break;
	case NodeType::COPIER:
	case NodeType::SPLITTER:
		new (&this->csn) CopierSplitterNode {};
		break;
	case NodeType::JOINER:
		new (&this->jn) JoinerNode {};
		break;
	default:
		unreachable("NetworkBlueprint::Node::Node");
	}
}

NetworkBlueprint::Node& NetworkBlueprint::Node::operator=(const Node& other) {
	if(this != &other) {
		type = other.type;
		numInput = other.numInput;
		numOutput = other.numOutput;
		switch(type) {
		case NodeType::LAYER:
			ln = LayerNode {};
			ln.bias = other.ln.bias;
			ln.inputID = other.ln.inputID;
			ln.outputID = other.ln.outputID;
			if(other.ln.blueprint) {
				ln.blueprint = other.ln.blueprint->clone();
			}
			ln.name = other.ln.name;
			ln.memreqs = other.ln.memreqs;
			break;
		case NodeType::COPIER:
		case NodeType::SPLITTER:
			csn = CopierSplitterNode {};
			csn.inputID = other.csn.inputID;
			csn.targets = vector<std::pair<size_t, size_t>>(other.csn.targets);
			break;
		case NodeType::JOINER:
			jn = JoinerNode {};
			jn.inputIDs = vector<size_t>(other.jn.inputIDs);
			jn.outputID = other.jn.outputID;
			break;
		default:
			unreachable("NetworkBlueprint::Node::operator=(const Node&)");
		}
	}
	return *this;
}

NetworkBlueprint::Node& NetworkBlueprint::Node::operator=(Node&& other) {
	if(this != &other) {
		type = other.type;
		numInput = other.numInput;
		numOutput = other.numOutput;
		switch(type) {
		case NodeType::LAYER:
			ln = std::move(other.ln);
			break;
		case NodeType::COPIER:
		case NodeType::SPLITTER:
			csn = std::move(other.csn);
			break;
		case NodeType::JOINER:
			jn = std::move(other.jn);
			break;
		default:
			unreachable("NetworkBlueprint::Node::operator=(Node&&)");
		}
	}
	return *this;
}

NetworkBlueprint::Node::Node(const Node& other) :
		Node(other.type) {
	this->operator=(other);
}

NetworkBlueprint::Node::Node(Node&& other) :
		Node(other.type) {
	this->operator=(std::move(other));
}

NetworkBlueprint::Node::~Node() {
	switch(type) {
	case NodeType::LAYER:
		this->ln.~LayerNode();
		break;
	case NodeType::COPIER:
	case NodeType::SPLITTER:
		this->csn.~CopierSplitterNode();
		break;
	case NodeType::JOINER:
		this->jn.~JoinerNode();
		break;
	}
}

NetworkBlueprint::CompiledNode& NetworkBlueprint::CompiledNode::operator=(const CompiledNode& other) {
	if(this != &other) {
		BB = other.BB;
		inputAt_where = other.inputAt_where;
		outputAt_where = other.outputAt_where;
		outputSlicing = other.outputSlicing;
		if(other.blueprint) {
			blueprint = other.blueprint->clone();
		} else {
			blueprint = nullptr;
		}
		layer = other.layer;
		biasAt = other.biasAt;
		inputAt_start = other.inputAt_start;
		outputAt_start = other.outputAt_start;
		inputSize = other.inputSize;
	}
	return *this;
}

/*
NetworkBlueprint::CompiledNode& NetworkBlueprint::CompiledNode::operator=(CompiledNode&& other) {
	if(this != &other) {
		BB = other.BB;
		inputAt_where = other.inputAt_where;
		outputAt_where = other.outputAt_where;
		outputSlicing = std::move(other.outputSlicing);
		blueprint = std::move(other.blueprint);
		layer = std::move(other.layer);
		biasAt = other.biasAt;
		inputAt_start = other.inputAt_start;
		outputAt_start = other.outputAt_start;
		inputSize = other.inputSize;
	}
	return *this;
}
*/

NetworkBlueprint::CompiledNode::CompiledNode(const CompiledNode& other) {
	this->operator=(other);
}

/*
NetworkBlueprint::CompiledNode::CompiledNode(CompiledNode&& other) {
	this->operator=(std::move(other));
}
*/

size_t NetworkBlueprint::getInputSize() const {
	return inputSize;
}

size_t NetworkBlueprint::getOutputSize() const {
	assertNetworkOutput(true);
	return outputSize;
}

size_t NetworkBlueprint::addLayer(const LayerBlueprint& layer, const std::string& name, size_t input_id, bool bias,
		size_t output_size) {
	assertNetworkOutput(false);
	Node new_node {NodeType::LAYER};
	if(!output_size) {
		throw std::invalid_argument(string("Output for ") + name + string(" set to 0"));
	}
	auto new_ID = getNextOutputID();
	new_node.numInput = setOutput(input_id, new_ID);
	new_node.numOutput = output_size;
	new_node.ln.bias = bias;
	new_node.ln.inputID = input_id ? input_id : NETWORK_IN_OUT;
	new_node.ln.outputID = NO_OUTPUT;
	assertIdentifier(name);
	if(name2ID.find(name) != name2ID.end()) {
		throw std::logic_error(string("Layer name ") + name + string(" is already in use"));
	}
	new_node.ln.name = name;
	new_node.ln.blueprint = layer.makeShaped(name, new_node.numInput + bias, output_size);
	new_node.ln.blueprint->getMemoryRequirements(new_node.ln.memreqs);

	name2ID.emplace(name, new_ID);
	nodes.push_back(std::move(new_node));
	return new_ID;
}

void NetworkBlueprint::addCopier(size_t input_id, vector<size_t>::iterator output_ids_start,
		vector<size_t>::iterator output_ids_end) {
	assertNetworkOutput(false);
	assert(output_ids_end >= output_ids_start);
	size_t num_outputs {static_cast<size_t>(output_ids_end - output_ids_start)};
	if(num_outputs < 2) {
		throw std::invalid_argument("A copier must have at least 2 outputs");
	} else if(num_outputs > MAX_NODES) {
		throw std::runtime_error("Too many outputs for a copier");
	}
	auto new_ID = getNextOutputID();
	Node new_node {NodeType::COPIER};
	new_node.numInput = setOutput(input_id, new_ID);
	assertSZMUL(new_node.numInput, num_outputs);
	new_node.numOutput = new_node.numInput * num_outputs;
	new_node.csn.inputID = input_id ? input_id : NETWORK_IN_OUT;
	new_node.csn.targets = std::vector(num_outputs, std::pair<size_t, size_t>(NO_OUTPUT, new_node.numInput));
	for(size_t i {0}; i < num_outputs; ++i) {
		*output_ids_start++ = (new_ID << 16) | i;
	}
	nodes.push_back(std::move(new_node));
}

void NetworkBlueprint::addSplitter(size_t input_id, vector<size_t>::iterator sizes_start,
		vector<size_t>::iterator sizes_end, vector<size_t>::iterator output_ids_start) {
	assertNetworkOutput(false);
	assert(sizes_end >= sizes_start);
	size_t num_outputs {static_cast<size_t>(sizes_end - sizes_start)};
	if(num_outputs < 2) {
		throw std::invalid_argument("A splitter must have at least 2 outputs");
	} else if(num_outputs > MAX_NODES) {
		throw std::runtime_error("Too many outputs for a splitter");
	}
	auto new_ID = getNextOutputID();
	Node new_node {NodeType::SPLITTER};
	new_node.numInput = setOutput(input_id, new_ID);
	new_node.numOutput = new_node.numInput;
	new_node.csn.inputID = input_id ? input_id : NETWORK_IN_OUT;
	new_node.csn.targets = std::vector<std::pair<size_t, size_t>> {};
	new_node.csn.targets.reserve(num_outputs);
	size_t totalSumOutputs {0};
	for(size_t i {0}; i < num_outputs; ++i) {
		if(new_node.numInput - totalSumOutputs < *sizes_start) {
			throw std::invalid_argument("Expected output of a splitter is bigger than its input");
		}
		totalSumOutputs += *sizes_start;
		new_node.csn.targets.emplace_back(NO_OUTPUT, *sizes_start);
		++sizes_start;
		*output_ids_start++ = (new_ID << 16) | i;
	}
	if(totalSumOutputs != new_node.numInput) {
		throw std::invalid_argument("Expected output of a splitter is less than its input");
	}
	nodes.push_back(std::move(new_node));
}

size_t NetworkBlueprint::addJoiner(vector<size_t>::iterator input_ids_start, vector<size_t>::iterator input_ids_end) {
	assertNetworkOutput(false);
	auto new_ID = getNextOutputID();
	assert(input_ids_end >= input_ids_start);
	size_t num_inputs {static_cast<size_t>(input_ids_end - input_ids_start)};
	if(num_inputs > MAX_NODES) {
		throw std::invalid_argument("A joiner has too many inputs");
	}
	Node new_node {NodeType::JOINER};
	new_node.numInput = 0;
	new_node.jn.inputIDs = std::vector<size_t> {};
	new_node.jn.inputIDs.reserve(num_inputs);
	new_node.jn.outputID = NO_OUTPUT;
	size_t idx_input {0};
	while(input_ids_start < input_ids_end) {
		if(!*input_ids_start) {
			throw std::invalid_argument("Network input cannot go directly into a joiner");
		}
		size_t next_input_size = setOutput(*input_ids_start, (new_ID << 16) | idx_input++);
		if(std::numeric_limits<size_t>::max() - next_input_size < new_node.numInput) {
			throw std::runtime_error("Total input size of a joiner is too big");
		}
		new_node.numInput += next_input_size;
		new_node.jn.inputIDs.push_back(*input_ids_start ? *input_ids_start : NETWORK_IN_OUT);
		++input_ids_start;
	}
	new_node.numOutput = new_node.numInput;
	nodes.push_back(std::move(new_node));
	return new_ID;
}

void NetworkBlueprint::addDiscarder(size_t input_id) {
	assertNetworkOutput(false);
	setOutput(input_id, DISCARD_OUTPUT);
}

void NetworkBlueprint::setNetworkOutput(size_t input_id) {
	assertNetworkOutput(false);
	if(!input_id) {
		throw std::logic_error("For technical reasons a network can't directly copy input to output");
	}

	// Check if the network is ready for compilation
	if(networkInputID == NO_OUTPUT) {
		throw std::logic_error("Trying to compile a network without an input");
	}
	for(size_t i {0}; i < nodes.size(); ++i) {
		size_t node_id {i + FIRST_FREE_OUTPUT};
		const auto& node = nodes.at(i);
		switch(node.type) {
		case NodeType::LAYER:
			if(node_id != input_id && node.ln.outputID == NO_OUTPUT) {
				throw std::logic_error(string("Layer ") + node.ln.name + std::string(" has unassigned output"));
			}
			break;
		case NodeType::JOINER:
			if(node_id != input_id && node.jn.outputID == NO_OUTPUT) {
				throw std::logic_error(string("Joiner ") + std::to_string(node_id) + string(" has unassigned output"));
			}
			break;
		case NodeType::COPIER:
		case NodeType::SPLITTER:
			for(size_t k {0}; k < node.csn.targets.size(); ++k) {
				size_t target_id {(node_id << 16) | k};
				if(target_id != input_id && node.csn.targets.at(k).first == NO_OUTPUT) {
					throw std::logic_error(
							string("Copier/splitter output ") + std::to_string(target_id) + string(" is unassigned"));
				}
			}
			break;
		default:
			unreachable("NetworkOutput::setNetworkOutput");
		}
	}

	// Set the output
	outputSize = setOutput(input_id, NETWORK_IN_OUT);
	networkOutputID = input_id;

	compileTrainable();
	compileForwardOnly();
}

void NetworkBlueprint::calculateMemoryRequirements(size_t& sz_weights, size_t& sz_pi_training, size_t& sz_pi_fwdonly) {
	assertNetworkOutput(true);
	sz_weights = totalMemoryRequirements.szPersistent;
	sz_pi_training = totalMemoryRequirements.szDeltas + totalMemoryRequirements.szInternalState
			+ realInputSizeTrainable * sizeof(FLOAT) + tempSizeTrainable * sizeof(FLOAT) + outputSize * sizeof(FLOAT);
	sz_pi_fwdonly = totalMemoryRequirements.szInternalState + realInputSizeFwdOnly * sizeof(FLOAT)
			+ tempSizeFwdOnly * sizeof(FLOAT) + outputSize * sizeof(FLOAT);
}

nlohmann::json NetworkBlueprint::serialize() {
	assertNetworkOutput(true);
	nlohmann::json result {
		{"name", networkName},
		{"input_size", inputSize},
		{"output", serializeIONName(networkOutputID)},
	};
	std::vector<nlohmann::json> serialization;
	serialization.reserve(nodes.size());
	for(size_t i {0}; i < nodes.size(); ++i) {
		const auto& node {nodes.at(i)};
		auto& snode = serialization.emplace_back();
		snode["name"] = serializeIONName(i + FIRST_FREE_OUTPUT);
		switch(node.type) {
		case NodeType::LAYER:
			snode["type"] = "layer";
			snode["input"] = serializeIONName(node.ln.inputID);
			snode["bias"] = node.ln.bias;
			snode["output_size"] = node.numOutput;
			node.ln.blueprint->serialize(snode);
			snode["discard_output"] = (node.ln.outputID == DISCARD_OUTPUT);
			break;
		case NodeType::COPIER:
			snode["type"] = "copier";
			snode["input"] = serializeIONName(node.csn.inputID);
			snode["num_outputs"] = node.csn.targets.size();
			snode["discard_outputs"] = serializeDiscardedOutputs(node.csn.targets);
			break;
		case NodeType::SPLITTER:
		{
			snode["type"] = "splitter";
			snode["input"] = serializeIONName(node.csn.inputID);
			std::vector<size_t> output_sizes(node.csn.targets.size());
			std::transform(node.csn.targets.begin(), node.csn.targets.end(), output_sizes.begin(), [](const std::pair<size_t, size_t>& x) -> size_t {
				return x.second;
			});
			snode["output_sizes"] = std::move(output_sizes);
			snode["discard_outputs"] = serializeDiscardedOutputs(node.csn.targets);
			break;
		}
		case NodeType::JOINER:
		{
			snode["type"] = "joiner";
			std::vector<string> inputs(node.jn.inputIDs.size());
			std::transform(node.jn.inputIDs.begin(), node.jn.inputIDs.end(), inputs.begin(), [this](size_t x) -> string {
				return serializeIONName(x);
			});
			snode["output_size"] = node.numOutput;
			snode["inputs"] = std::move(inputs);
			snode["discard_output"] = (node.jn.outputID == DISCARD_OUTPUT);
			break;
		}
		default:
			unreachable("NetworkBlueprint::serialize");
		}
	}
	result["nodes"] = std::move(serialization);
	return result;
}

string NetworkBlueprint::serializeAsString() {
	auto result = serialize();
	return result.dump();
}

void NetworkBlueprint::assertNetworkOutput(bool is_set) const {
	if(is_set) {
		if(networkOutputID == NO_OUTPUT) {
			throw std::logic_error("Network output has not yet been set");
		}
	} else {
		if(networkOutputID != NO_OUTPUT) {
			throw std::logic_error("Network output has already been set");
		}
	}
}

string NetworkBlueprint::serializeIONName(size_t ion_id) const {
	if(ion_id > 0xFFFF) {
		return string("#") + std::to_string(ion_id >> 16) + string(":") + std::to_string((ion_id & 0xFFFF) + 1);
	} else if(ion_id == NETWORK_IN_OUT) {
		return string("@network");
	} else {
		assert(ion_id >= FIRST_FREE_OUTPUT);
		const auto& node {nodes.at(ion_id - FIRST_FREE_OUTPUT)};
		if(node.type == NodeType::LAYER) {
			return node.ln.name;
		} else {
			return string("#") + std::to_string(ion_id);
		}
	}
}

std::vector<size_t> NetworkBlueprint::serializeDiscardedOutputs(const vector<std::pair<size_t, size_t>>& targets) const {
	std::vector<size_t> result;
	for(size_t i {0}; i < targets.size(); ++i) {
		if(targets.at(i).first == DISCARD_OUTPUT) {
			result.push_back(i + 1);
		}
	}
	return result;
}

size_t NetworkBlueprint::getCSOutput(size_t input_id, size_t& target_index, bool must_be_free) {
	if(input_id < FIRST_FREE_OUTPUT) {
		throw std::invalid_argument(std::to_string(input_id) + string(" is not a valid input ID"));
	}
	if(input_id <= 0xFFFF) {
		return 0;
	} else {
		size_t idx_input_node = (input_id >> 16) - FIRST_FREE_OUTPUT;
		size_t expected_target_index = input_id & 0xFFFF;
		if(nodes.size() > idx_input_node) {
			auto& input_node = nodes.at(idx_input_node);
			if((input_node.type == NodeType::COPIER || input_node.type == NodeType::SPLITTER)
					&& input_node.csn.targets.size() > expected_target_index) {
				if(must_be_free && input_node.csn.targets.at(expected_target_index).first != NO_OUTPUT) {
					throw std::logic_error(
							string("Output ") + std::to_string(input_id) + string(" is already assigned to an input"));
				} else {
					target_index = expected_target_index;
					return idx_input_node + FIRST_FREE_OUTPUT;
				}
			}
		}
	}
	// If we land here, something went wrong
	throw std::invalid_argument(string("Input ID not (yet) assigned: ") + std::to_string(input_id));
}

size_t NetworkBlueprint::getNextOutputID() const {
	if(nodes.size() == MAX_NODES) {
		throw std::runtime_error("The neural network has too many layers/nodes");
	}
	return nodes.size() + FIRST_FREE_OUTPUT;
}

size_t NetworkBlueprint::setOutput(size_t input_id, size_t new_output_id) {
	if(!input_id) {
		if(networkInputID != NO_OUTPUT) {
			throw std::logic_error("Network input is already set");
		} else if(new_output_id == DISCARD_OUTPUT) {
			throw std::logic_error("Entire network input cannot be discarded");
		} else {
			networkInputID = new_output_id;
			return inputSize;
		}
	} else {
		assert(input_id >= FIRST_FREE_OUTPUT);
		size_t target_index {0};
		size_t input_node_ID = getCSOutput(input_id, target_index, true);
		if(input_node_ID) {
			// Copier or splitter
			auto& target = nodes.at(input_node_ID - FIRST_FREE_OUTPUT).csn.targets.at(target_index);
			target.first = new_output_id;
			return target.second;
		} else {
			// Layer or joiner
			bool output_was_free {false};
			auto& input_node = nodes.at(input_id - FIRST_FREE_OUTPUT);
			switch(input_node.type) {
			case NodeType::LAYER:
				if(input_node.ln.outputID == NO_OUTPUT) {
					input_node.ln.outputID = new_output_id;
					output_was_free = true;
				}
				break;
			case NodeType::JOINER:
				if(input_node.jn.outputID == NO_OUTPUT) {
					input_node.jn.outputID = new_output_id;
					output_was_free = true;
				}
				break;
			default:
				unreachable("NetworkBlueprint::setOutput");
			}
			if(output_was_free) {
				return input_node.numOutput;
			} else {
				throw std::logic_error(
						string("Output ") + std::to_string(input_id) + string(" is already assigned to an input"));
			}
		}
	}
}

size_t NetworkBlueprint::getAllocationIndex(size_t input_id, size_t& idx_allocation) {
	if(input_id > 0xFFFF) {
		idx_allocation = (input_id & 0xFFFF);
		return (input_id >> 16) - FIRST_FREE_OUTPUT;
	} else {
		idx_allocation = 0;
		return input_id - FIRST_FREE_OUTPUT;
	}
}

NetworkBlueprint::NodeAllocation& NetworkBlueprint::findAllocation(vector<vector<NodeAllocation>>& allocations,
		size_t input_id) {
	size_t idx_allocation {0};
	size_t idx_node = getAllocationIndex(input_id, idx_allocation);
	return allocations.at(idx_node).at(idx_allocation);
}

void NetworkBlueprint::startCompilingJoiner(size_t& storage_size, vector<vector<NodeAllocation>>& allocations,
		size_t idx_joiner) {
	auto& allocs = allocations.at(idx_joiner);
	if(allocs.empty()) {
		const auto& node = nodes.at(idx_joiner);
		assert(node.type == NodeType::JOINER);
		const auto& inputIDs = node.jn.inputIDs;
		allocs = std::vector<NodeAllocation>(inputIDs.size() + 1);
		auto& alloc_main = allocs.at(0);
		if(node.jn.outputID == NETWORK_IN_OUT) {
			alloc_main.where = StorageType::NETOUT;
			alloc_main.index = 0;
		} else {
			alloc_main.where = StorageType::TEMP;
			++storage_size;
			alloc_main.index = storage_size;
			if(node.jn.outputID != DISCARD_OUTPUT) {
				assert(node.jn.outputID >= FIRST_FREE_OUTPUT);
				const auto& output_node = nodes.at(ID2Index(node.jn.outputID));
				if(output_node.type == NodeType::SPLITTER) {
					// Allocate memory for biases between targets
					storage_size += output_node.csn.targets.size() - 1;
				}
			}
		}
		alloc_main.remainingJoinerInputsP1 = inputIDs.size() + 1;
		size_t prev_input_size {0};
		for(size_t i {0}; i < inputIDs.size(); ++i) {
			size_t input_id = inputIDs.at(i);
			size_t idx_allocation {0};
			size_t idx_node {getAllocationIndex(input_id, idx_allocation)};
			size_t input_size;
			if(inputIDs.at(i) > 0xFFFF) {
				// idx_allocation will match the target number in the input node
				input_size = nodes.at(idx_node).csn.targets.at(idx_allocation).second;
			} else {
				input_size = nodes.at(idx_node).numOutput;
			}
			allocs.at(i + 1).where = StorageType::UNUSED; // As a placeholder to mean that the input is not yet available
			allocs.at(i + 1).index = allocs.at(i).index + prev_input_size;
			storage_size += input_size;
			prev_input_size = input_size;
		}
	}
	// If the joiner is already allocated, do nothing
}

bool NetworkBlueprint::updateJoiner(size_t& storage_size, vector<CompiledNode> compilation,
		vector<vector<NodeAllocation>>& allocations, size_t idx_joiner) {
	startCompilingJoiner(storage_size, allocations, idx_joiner);
	const auto& node = nodes.at(idx_joiner);
	auto& allocs = allocations.at(idx_joiner);
	auto& alloc_main = allocs.at(0);
	if(!alloc_main.remainingJoinerInputsP1) {
		return false;
	} else {
		const auto& inputIDs = node.jn.inputIDs;
		for(size_t i {0}; i < inputIDs.size(); ++i) {
			auto& alloc_part = allocs.at(i + 1);
			if(alloc_part.where == StorageType::UNUSED) {
				size_t idx_input_allocation {0};
				size_t idx_input_node = getAllocationIndex(inputIDs.at(i), idx_input_allocation);
				const auto& input_node_alloc = allocations.at(idx_input_allocation);
				size_t input_size;
				if(inputIDs.at(i) > 0xFFFF) {
					// idx_allocation will match the target number in the input node
					input_size = nodes.at(idx_input_node).csn.targets.at(idx_input_allocation).second;
				} else {
					input_size = nodes.at(idx_input_node).numOutput;
				}

				if(!input_node_alloc.empty() && !input_node_alloc.at(0).remainingJoinerInputsP1) {
					// Unless it's a joiner, if it's allocated, it's available,
					// so copy it into the joiner. If it is a joiner and not yet
					// complete, it will eventually be processed once completed.
					const auto& input_allocation = input_node_alloc.at(idx_input_allocation);
					CompiledNode cnode {};
					cnode.inputAt_where = input_allocation.where;
					cnode.inputAt_start = input_allocation.index;
					cnode.outputAt_where = alloc_main.where;
					cnode.outputAt_start = allocs.at(i + 1).index;
					cnode.inputSize = input_size;
					--alloc_main.remainingJoinerInputsP1;
					compilation.push_back(std::move(cnode));
					alloc_part.where = alloc_main.where;
				}
			}
		}
		return (alloc_main.remainingJoinerInputsP1 == 1);
	}
}

void NetworkBlueprint::attachJoinerInput(vector<vector<NodeAllocation>>& allocations, std::list<size_t>& stack,
		size_t joiner_input_id, CompiledNode& cnode) {
	assert(joiner_input_id > 0xFFFF);
	size_t idx_input_alloc {0};
	size_t idx_joiner {getAllocationIndex(joiner_input_id, idx_input_alloc)};
	auto& allocs = allocations.at(idx_joiner);
	auto& alloc_main = allocs.at(0);
	auto& alloc_input = allocs.at(idx_input_alloc + 1);
	assert(alloc_input.where == StorageType::UNUSED);
	alloc_input.where = alloc_main.where;
	cnode.outputAt_where = alloc_main.where;
	cnode.outputAt_start = alloc_input.index;
	assert(alloc_main.remainingJoinerInputsP1 > 1);
	--alloc_main.remainingJoinerInputsP1;
	if(alloc_main.remainingJoinerInputsP1 == 1) {
		stack.push_back(joiner_input_id >> 16);
	}
}

size_t NetworkBlueprint::compileSplitOutput(vector<vector<NodeAllocation>>& allocations, std::list<size_t>& stack,
		size_t idx_splitter, vector<size_t>& output_slicing, StorageType startAt_where, size_t startAt_index) {
	auto& node = nodes.at(idx_splitter);
	assert(node.type == NodeType::SPLITTER);
	auto& allocs = allocations.at(idx_splitter);
	assert(allocs.empty());
	size_t numtgt = node.csn.targets.size();
	allocs = std::vector<NodeAllocation>(numtgt);
	size_t shift = node.numInput;
	for(size_t i {numtgt - 1}; i > 0; --i) {
		size_t sz {node.csn.targets.at(i).second};
		shift -= sz;
		allocs.at(i) = NodeAllocation(startAt_where, startAt_index + shift + i);
		output_slicing.push_back(shift);
		output_slicing.push_back(sz);
		output_slicing.push_back(shift + i);
		size_t target {node.csn.targets.at(i).first};
		if(target == DISCARD_OUTPUT) {
			output_slicing.push_back(0);
		} else {
			output_slicing.push_back(1);
			stack.push_back(target);
		}
	}
	size_t target {node.csn.targets.at(0).first};
	if(target != DISCARD_OUTPUT) {
		stack.push_back(target);
	}
	allocs.at(0) = NodeAllocation(startAt_where, startAt_index);
	return numtgt - 1;
}

void NetworkBlueprint::compileCopierTrainable(size_t& storage_size, vector<CompiledNode> compilation,
		vector<vector<NodeAllocation>>& allocations, std::list<size_t>& stack, size_t idx_copier,
		StorageType inputAt_where, size_t inputAt_start) {
	auto& node = nodes.at(idx_copier);
	assert(node.type == NodeType::COPIER);
	auto& allocs = allocations.at(idx_copier);
	assert(allocs.empty());
	auto& targets = node.csn.targets;

	// Allocate all joiners before we fill our own allocations
	for(auto it = targets.begin(); it != targets.end(); ++it) {
		size_t target = it->first;
		if(target > 0xFFFF) {
			bool joiner_done {updateJoiner(storage_size, compilation, allocations, ID2Index(target))};
			// This output hasn't been compiled yet
			assert(!joiner_done);
		}
	}

	// Start our own allocations
	allocs = std::vector<NodeAllocation>(targets.size());
	// For code simplicity we copy even the first target. TODO: do not if unnecessary
	for(size_t i {0}; i < targets.size(); ++i) {
		size_t tgt = targets.at(i).first;
		auto& tgt_alloc = allocs.at(i);
		CompiledNode cnode {};
		// The last node copies derivatives on backward runs
		// others add them
		cnode.BB = (i < targets.size() - 1 ? BackwardBehavior::ADD_DERIVS : BackwardBehavior::CALL_OR_COPY);
		cnode.inputAt_where = inputAt_where;
		cnode.inputAt_start = inputAt_start;
		cnode.inputSize = node.numInput;
		if(tgt == DISCARD_OUTPUT) {
			// Fake slicing to get zero derivatives
			cnode.outputSlicing.push_back(0);
			cnode.outputSlicing.push_back(node.numInput);
			cnode.outputSlicing.push_back(0);
			cnode.outputSlicing.push_back(0);
		} else if(tgt == NETWORK_IN_OUT) {
			// Allocate the target right in the output
			tgt_alloc.where = StorageType::NETOUT;
			tgt_alloc.index = 0;
		} else if(tgt > 0xFFFF) {
			size_t idx_joiner_alloc {0};
			size_t idx_joiner {getAllocationIndex(tgt, idx_joiner_alloc)};
			auto& joiner_alloc = allocations.at(idx_joiner).at(idx_joiner_alloc + 1);
			// ^ +1 because getAllocationIndex returns values for inputs
			auto& joiner_main = allocations.at(idx_joiner).at(0);
			joiner_alloc.where = joiner_main.where;
			tgt_alloc.where = joiner_alloc.where;
			tgt_alloc.index = joiner_alloc.index;
			assert(joiner_main.remainingJoinerInputsP1 > 0);
			--joiner_main.remainingJoinerInputsP1;
			if(joiner_main.remainingJoinerInputsP1 == 1) {
				stack.push_back(tgt);
			}
		} else {
			assert(tgt >= FIRST_FREE_OUTPUT);
			tgt_alloc.where = StorageType::TEMP;
			tgt_alloc.index = storage_size + 1;
			storage_size += 1 + node.numInput;
			size_t idx_output {ID2Index(tgt)};
			auto& output_node = nodes.at(idx_output);
			if(output_node.type == NodeType::SPLITTER) {
				storage_size += compileSplitOutput(allocations, stack, idx_output, cnode.outputSlicing, tgt_alloc.where,
						tgt_alloc.index);
			} else {
				stack.push_back(tgt);
			}
		}
		cnode.outputAt_where = tgt_alloc.where;
		cnode.outputAt_start = tgt_alloc.index;
		compilation.push_back(std::move(cnode));
	}
}

void NetworkBlueprint::compileTrainable() {
	// This is long, I know. Bear with me...
	// Firstly, we declare the state
	size_t real_input_size {inputSize + 1};
	size_t storage_size {0};
	/*
	 * `allocations` contain allocation data for every original node, as follows:
	 * - For a LAYER: a single element indicating where the output is
	 * - For a COPIER: one element per output indicating where each copy is
	 * - For a SPLITER: one element per output indicating where each part is
	 * - For a JOINER: element 0 indicates where the output is. Elements 1 to (number of inputs)
	 *     indicate where the space for each input (by its 1-based index) is allocated.
	 *     .remainingJoineInputs is set on the 0th element: if 0, the joiner has already been compiled;
	 *     if 1, the joiner is ready to compile; if N > 1, N - 1 inputs still need to be processed.
	 *
	 * In all cases potential bias inputs don't count. There should be place for them at (index - 1).
	 * Joiners do not have biases between parts.
	 */
	vector<vector<NodeAllocation>> allocations(nodes.size());
	std::list<size_t> stack {};
	compilationTrainable = std::make_unique<vector<CompiledNode>>();
	compilationTrainable->reserve(nodes.size());

	// Secondly, we process the input
	size_t idx_input_node {ID2Index(networkInputID)};
	auto& input_node = nodes.at(idx_input_node);
	switch(input_node.type) {
	case NodeType::LAYER: {
		CompiledNode cnode {};
		cnode.inputAt_where = StorageType::NETIN;
		cnode.biasAt = 0;
		cnode.inputAt_start = (input_node.ln.bias ? 0 : 1);
		cnode.blueprint = input_node.ln.blueprint->clone();
		if(input_node.ln.outputID == NETWORK_IN_OUT) {
			// Our network consists of one node
			cnode.outputAt_where = StorageType::NETOUT;
			cnode.outputAt_start = 0;
			// Allocations shouldn't matter
		} else {
			// It cannot be a discarder or a joiner, else the network will have not output
			size_t idx_input_node_output {ID2Index(input_node.ln.outputID)};
			auto& input_node_output = nodes.at(idx_input_node_output);
			cnode.outputAt_where = StorageType::TEMP;
			cnode.outputAt_start = 1;
			allocations.at(idx_input_node).push_back(NodeAllocation(StorageType::TEMP, 1));
			storage_size += input_node.numOutput + 1;
			if(input_node_output.type == NodeType::SPLITTER) {
				// Split in place
				storage_size += compileSplitOutput(allocations, stack, idx_input_node_output, cnode.outputSlicing,
						StorageType::TEMP, 1);
				// The call will have pushed the splitter's outputs onto the stack
			} else {
				stack.push_back(input_node.ln.outputID);
			}
		}
		compilationTrainable->push_back(std::move(cnode));
		break;
	}
	case NodeType::SPLITTER: {
		// Split input in-place
		CompiledNode cnode {};
		cnode.outputAt_where = StorageType::NETIN;
		cnode.outputAt_start = 1;
		real_input_size += compileSplitOutput(allocations, stack, idx_input_node, cnode.outputSlicing,
				StorageType::NETIN, 1);
		compilationTrainable->push_back(std::move(cnode));
		break;
	}
	case NodeType::COPIER:
		compileCopierTrainable(storage_size, *compilationTrainable, allocations, stack, idx_input_node,
				StorageType::NETIN, 1);
		break;
	default:
		unreachable("NetworkBlueprint::compileTrainable (input)");
	}

	// Thirdly, we compile the rest of the stack
	while(!stack.empty()) {
		size_t next_id {stack.front()};
		stack.pop_front();
		if(next_id == NETWORK_IN_OUT) {
			// The output has become available
			size_t idx_allocation {0};
			size_t idx_node {getAllocationIndex(networkOutputID, idx_allocation)};
			auto& alloc = allocations.at(idx_node).at(idx_allocation);
			auto& output_node = nodes.at(idx_node);
			CompiledNode cnode {};
			cnode.inputAt_where = alloc.where;
			cnode.inputAt_start = alloc.index;
			cnode.outputAt_where = StorageType::NETOUT;
			cnode.outputAt_start = 0;
			switch(output_node.type) {
			case NodeType::LAYER:
			case NodeType::JOINER:
				cnode.inputSize = output_node.numOutput;
				break;
			case NodeType::COPIER:
			case NodeType::SPLITTER:
				cnode.inputSize = output_node.csn.targets.at(idx_allocation).second;
				break;
			default:
				unreachable("NetworkBlueprint::compileTrainable (NETWORK_IN_OUT)");
			}
			compilationTrainable->push_back(std::move(cnode));
			stack.clear();	// Don't care about any other nodes
		} else {
			assert(next_id >= FIRST_FREE_OUTPUT);
			// We do not actually care about the particular target
			size_t idx_next_node {ID2Index(next_id)};
			auto& next_node = nodes.at(idx_next_node);
			switch(next_node.type) {
			case NodeType::LAYER: {
				auto& next_alloc = allocations.at(idx_next_node);
				assert(next_alloc.empty());
				CompiledNode cnode {};
				auto& input_alloc = findAllocation(allocations, next_node.ln.inputID);
				cnode.inputAt_where = input_alloc.where;
				cnode.biasAt = input_alloc.index - 1;
				cnode.inputAt_start = (next_node.ln.bias ? input_alloc.index - 1 : input_alloc.index);
				cnode.blueprint = next_node.ln.blueprint->clone();
				if(next_node.ln.outputID == NETWORK_IN_OUT) {
					cnode.outputAt_where = StorageType::NETOUT;
					cnode.outputAt_start = 0;
					stack.clear();
				} else if(next_node.ln.outputID > 0xFFFF) {
					updateJoiner(storage_size, *compilationTrainable, allocations, ID2Index(next_node.ln.outputID));
					attachJoinerInput(allocations, stack, next_node.ln.outputID, cnode);
				} else {
					cnode.outputAt_where = StorageType::TEMP;
					cnode.outputAt_start = storage_size + 1;
					storage_size += 1 + next_node.numOutput;
					// To improve cache locality we should process the output ASAP
					stack.push_front(next_node.ln.outputID);
				}
				next_alloc.push_back(NodeAllocation(cnode.outputAt_where, cnode.outputAt_start));
				compilationTrainable->push_back(std::move(cnode));
				break;
			}
			case NodeType::COPIER: {
				auto& input_alloc = findAllocation(allocations, next_node.csn.inputID);
				compileCopierTrainable(storage_size, *compilationTrainable, allocations, stack, idx_next_node,
						input_alloc.where, input_alloc.index);
				break;
			}
			case NodeType::SPLITTER: {
				// If a splitter landed on the stack, we'll need to copy the data
				CompiledNode cnode {};
				auto& input_alloc = findAllocation(allocations, next_node.csn.inputID);
				cnode.inputAt_where = input_alloc.where;
				cnode.inputAt_start = input_alloc.index;
				cnode.outputAt_where = StorageType::TEMP;
				cnode.outputAt_start = storage_size + 1;
				cnode.inputSize = next_node.numInput;
				storage_size += 1 + next_node.numInput
						+ compileSplitOutput(allocations, stack, idx_next_node, cnode.outputSlicing,
								cnode.outputAt_where, cnode.outputAt_start);
				compilationTrainable->push_back(std::move(cnode));
				break;
			}
			case NodeType::JOINER: {
				if(updateJoiner(storage_size, *compilationTrainable, allocations, idx_next_node)) {
					stack.push_front(next_node.jn.outputID);
				}
				break;
			}
			default:
				unreachable("NetworkBlueprint::compileTrainable (stack)");
			}
		}
	}

	// Finally, we compute the memory requirements
	compilationTrainable->shrink_to_fit();
	realInputSizeTrainable = real_input_size;
	tempSizeTrainable = storage_size;
	for(const auto& node : nodes) {
		if(node.type == NodeType::LAYER) {
			totalMemoryRequirements.numTempStateBackward = std::max(totalMemoryRequirements.numTempStateBackward,
					node.ln.memreqs.numTempStateBackward);
			totalMemoryRequirements.numTempStateForward = std::max(totalMemoryRequirements.numTempStateForward,
					node.ln.memreqs.numTempStateForward);
			totalMemoryRequirements.szPersistent += node.ln.memreqs.szPersistent;
			totalMemoryRequirements.szDeltas += node.ln.memreqs.szDeltas;
			totalMemoryRequirements.szInternalState += node.ln.memreqs.szInternalState;
		}
	}
}

void NetworkBlueprint::compileForwardOnly() {
	// TODO: a forward-only version can be much more optimized
	compilationFwdOnly = std::make_unique<vector<CompiledNode>>(compilationTrainable->size());
	std::copy(compilationTrainable->begin(), compilationTrainable->end(), compilationFwdOnly->begin());
	realInputSizeFwdOnly = realInputSizeTrainable;
	tempSizeFwdOnly = tempSizeTrainable;
}

shared_ptr<NetworkPool> NetworkBlueprint::createPool(H5::Group* weights_from) {
	return std::shared_ptr<NetworkPool>(new NetworkPool(networkName, weights_from, "SERIALIZATION NOT IMPLEMENTED YET",
			inputSize, outputSize, realInputSizeTrainable, realInputSizeFwdOnly, tempSizeTrainable,
			tempSizeFwdOnly, std::max(totalMemoryRequirements.numTempStateForward, totalMemoryRequirements.numTempStateBackward),
			totalMemoryRequirements.numTempStateForward, *compilationTrainable, *compilationFwdOnly));
}

NetworkPool::NetworkPool(const string& network_name,
		H5::Group* weights_from,
		const string& blueprint_serialization,
		size_t input_size, size_t output_size,
		size_t real_input_size_trainable,
		size_t real_input_size_fwdonly,
		size_t num_temp_trainable, size_t num_temp_fwd_only,
		size_t num_scratch_trainable, size_t num_scratch_fwd_only,
		const std::vector<NetworkBlueprint::CompiledNode>& nodes_trainable,
		const std::vector<NetworkBlueprint::CompiledNode>& nodes_fwdonly) :
				networkName(network_name),
		blueprintSerialization(blueprint_serialization),
		inputSize(input_size), outputSize(output_size), realInputSizeTrainable(real_input_size_trainable),
		realInputSizeFwdOnly(real_input_size_fwdonly),
		numTempTrainable(num_temp_trainable), numTempFwdOnly(
				num_temp_fwd_only), numScratchTrainable(num_scratch_trainable), numScratchFwdOnly(num_scratch_fwd_only), nodesTrainable(std::vector<NetworkBlueprint::CompiledNode>(nodes_trainable.size())),
				nodesFwdOnly(std::vector<NetworkBlueprint::CompiledNode>(nodes_fwdonly.size())), name2pool(std::unordered_map<string, shared_ptr<LayerPool>>(1)) {
	for(size_t i {0}; i < nodes_trainable.size(); ++i) {
		const auto& node = nodes_trainable.at(i);
		auto& out_node = this->nodesTrainable.at(i);
		out_node = node;
		if(node.blueprint) {
			string name;
			size_t tmp1, tmp2;
			node.blueprint->getShape(name, tmp1, tmp2);
			assert(name2pool.find(name) == name2pool.end());
			out_node.layer = node.blueprint->createPool(network_name, weights_from);
			name2pool.emplace(name, out_node.layer);
		}
	}

	for(size_t i {0}; i < nodes_fwdonly.size(); ++i) {
		const auto& node = nodes_fwdonly.at(i);
		auto& out_node = this->nodesTrainable.at(i);
		out_node = node;
		if(node.blueprint) {
			string name;
			size_t tmp1, tmp2;
			node.blueprint->getShape(name, tmp1, tmp2);
			auto it = name2pool.find(name);
			assert(it != name2pool.end());
			out_node.layer = it->second;
		}
	}
}

void NetworkPool::initializeWeights(array<uint64_t, 4>& seed) {
	// To preserve the order we iterator over nodes, not the name map
	for(auto& cnode : this->nodesTrainable) {
		if(cnode.layer) {
			cnode.layer->initializeWeights(seed);
		}
	}
}

shared_ptr<LayerPool> NetworkPool::getLayerPool(const std::string& name) const {
	auto it = name2pool.find(name);
	if(it == name2pool.end()) {
		throw std::invalid_argument(string("No layer named ") + name);
	} else {
		return it->second;
	}
}

unique_ptr<NetworkInstance> NetworkPool::createInstance(bool trainable) {
	auto& compilation = (trainable ? nodesTrainable : nodesFwdOnly);
	vector<FLOAT> input(trainable ? realInputSizeTrainable : realInputSizeFwdOnly);
	input.shrink_to_fit();
	vector<FLOAT> output(outputSize);
	output.shrink_to_fit();
	vector<FLOAT> temp(trainable ? numTempTrainable : numTempFwdOnly);
	temp.shrink_to_fit();
	vector<FLOAT> scratch(trainable ? numScratchTrainable : numScratchFwdOnly);
	scratch.shrink_to_fit();
	vector<NetworkInstance::Node> result_nodes(compilation.size());
	result_nodes.shrink_to_fit();

	auto translate = [&](NetworkBlueprint::StorageType type) -> FLOAT* {
		switch(type) {
		case NetworkBlueprint::StorageType::NETIN:
			return input.data();
		case NetworkBlueprint::StorageType::NETOUT:
			return output.data();
		case NetworkBlueprint::StorageType::TEMP:
			return temp.data();
		default:
			unreachable("NetworkPool::createInstance::translate");
		}
	};

	for(size_t i {0}; i < compilation.size(); ++i) {
		const auto& cnode = compilation.at(i);
		auto& result = result_nodes.at(i);
		result.outputSlicing = vector<size_t>(cnode.outputSlicing);
		result.backwardAddDerivs = (cnode.BB == NetworkBlueprint::BackwardBehavior::ADD_DERIVS);
		auto input_base = translate(cnode.inputAt_where);
		result.biasAt = input_base + cnode.biasAt;
		result.inputAt = input_base + cnode.inputAt_start;
		if(cnode.outputAt_where == NetworkBlueprint::StorageType::UNUSED) {
			result.outputAt = nullptr;
		} else {
			result.outputAt = translate(cnode.outputAt_where) + cnode.outputAt_start;
		}
		if(cnode.layer) {
			result.layer = cnode.layer->createInstance(trainable, result.inputAt, result.outputAt, scratch.data());
		}
		result.inputSize = cnode.inputSize;
	}

	return unique_ptr<NetworkInstance>(new NetworkInstance(trainable, inputSize, outputSize, std::move(input), std::move(output),
			std::move(temp), std::move(scratch), std::move(result_nodes)));
}

NetworkInstance::NetworkInstance(bool trainable, size_t input_size, size_t output_size,
		vector<FLOAT>&& input,
			vector<FLOAT>&& output, vector<FLOAT>&& temp,
			vector<FLOAT>&& scratch, vector<Node>&& nodes) : trainable(trainable), inputSize(input_size), outputSize(output_size), nodes(std::move(nodes)),
					input(std::move(input)), output(std::move(output)), temp(std::move(temp)), scratch(std::move(scratch)) {
}

size_t NetworkInstance::getInput(FLOAT** ptr) {
	if(ptr) {
		*ptr = input.data() + 1;	// +1 because bias
	}
	return inputSize;
}

size_t NetworkInstance::getOutput(FLOAT** ptr) {
	if(ptr) {
		*ptr = output.data();
	}
	return outputSize;
}

bool NetworkInstance::isTrainable() const {
	return trainable;
}

void NetworkInstance::forward() {
	for(auto& node : nodes) {
		FLOAT* output_at;

		// Produce output from input
		if(node.layer) {
			*node.biasAt = 1;
			node.layer->forward();
			output_at = node.outputAt;
		} else if(node.outputAt) {
			output_at = node.outputAt;
			std::memmove(reinterpret_cast<char*>(output_at), reinterpret_cast<char*>(node.inputAt),
					node.inputSize * sizeof(FLOAT));
		} else {
			output_at = node.inputAt;
		}

		// Slice
		for(size_t i {0}; i < node.outputSlicing.size(); i += 4) {
			assert(node.outputSlicing.size() >= i + 3);
			std::memmove(reinterpret_cast<char*>(output_at + node.outputSlicing[i + 2]),
					reinterpret_cast<char*>(output_at + node.outputSlicing[i]),
					sizeof(FLOAT) * node.outputSlicing[i + 1]);
			// node.outputSlicing.at[i + 3] is only used during backward runs
		}
	}
}

void NetworkInstance::backward(FLOAT lrate, const FLOAT* expected) {
	auto out = output.data();
	// TODO: maybe add other loss functions
	std::transform(const_cast<FLOAT*>(expected), const_cast<FLOAT*>(expected) + outputSize, out, out, [](FLOAT a, FLOAT b) -> FLOAT {
		return b - a;
	});
	backwardGradient(lrate, out);
}

void NetworkInstance::backwardGradient(FLOAT lrate, const FLOAT* derivs) {
	auto out = output.data();
	if(derivs != out) {
		std::memmove(reinterpret_cast<char*>(out), reinterpret_cast<const char*>(derivs),
				outputSize * sizeof(FLOAT));
	}
	for(auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
		auto& node = *it;

		auto output_at = (node.outputAt ? node.outputAt : node.inputAt);

		// Unslice
		assert(node.outputSlicing.size() % 4 == 0);
		for(size_t i {node.outputSlicing.size()}; i > 0; i -= 4) {
			if(!node.outputSlicing[i - 1]) {
				// A zero there is used when a discarder requires faking zero gradient
				std::fill<FLOAT*, FLOAT>(output_at + node.outputSlicing[i - 4],
						output_at + node.outputSlicing[i - 4] + node.outputSlicing[i - 3],
						0);
			} else {
				std::memmove(reinterpret_cast<char*>(output_at + node.outputSlicing[i - 2]),
						reinterpret_cast<char*>(output_at + node.outputSlicing[i - 4]),
						sizeof(FLOAT) * node.outputSlicing[i - 3]);
			}
		}

		// Do the backward run
		if(node.layer) {
			node.layer->backward(lrate);
		} else if(node.outputAt) {
			if(node.backwardAddDerivs) {
				clamp_axpy(1, node.outputAt, node.inputSize, node.inputAt);
			} else {
				std::memmove(reinterpret_cast<char*>(node.inputAt), reinterpret_cast<char*>(node.outputAt),
						sizeof(FLOAT) * node.inputSize);
			}
		}
	}
}

void NetworkInstance::updateWeights(FLOAT proportion) {
	for(auto& node : nodes) {
		if(node.layer) {
			node.layer->updateWeights(proportion);
		}
	}
}

void NetworkInstance::resetState() {
	for(auto& node : nodes) {
		if(node.layer) {
			node.layer->resetState();
		}
	}
}
}
