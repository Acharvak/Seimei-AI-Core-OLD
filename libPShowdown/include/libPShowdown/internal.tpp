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
#ifndef SEIMEI_LIBPSHOWDOWN_HPP_
#include <libPShowdown.hpp>
#endif

#include <cassert>
#include <stdexcept>
#include <variant>

namespace LIBPSHOWDOWN_NAMESPACE {
namespace {
// We need this before including tries.tpp
enum class ServerReply {
	EMPTY_LINE,

	// Player identifiers
	p1,
	p2,
	p3,
	p4,

	// === Commands before or after "start"

	// One string argument: "singles", "doubles", "triples", "multi" or "free-for-all"
	gametype,
	// One integer argument: generation number
	gen,
	// Arguments: player ID (string)|name (string)|avatar ID (string)|rating (integer or blank)
	player,
	// Argument till end of line: request data (JSON)
	request,
	// One string argument: active rule clause, e.g. "Sleep Clause Mod: Limit one foe put to sleep"
	rule,
	// No arguments. Signals that side information has been updated, next reply will normally be the player ID
	sideupdate,
	// One string argument: player ID. Signals that two messages will follow:
	// one for a particular player, one for everyone else. Only happens during
	// DirectSimConnection.
	split,
	// No arguments. Supposed to mean that the battle or team preview has started, but unless
	// "teampreview" has been sent, one should wait for "turn"
	start,
	// Two arguments: player ID (string) and an integer denoting the size of their battling team
	teamsize,
	// No arguments. Seems to be only useful for logging
	update,

	// === Team Preview (always before "start")

	// No arguments. Start of team preview.
	clearpoke,
	// Arguments: player ID, Pokémon details, empty field or "item". Preview Pokémon.
	poke,
	// No arguments. Team preview is ready.
	teampreview,

	// === Commands after "start"

	// Arguments: position|Pokémon details|health
	drag,
	// The actual header is "switch". Arguments like "drag". Note that it may be sent
	// in random battles to indicate the player's automatically chosen lead.
	switch_position,
	// One integer argument: turn number, starting with 1
	turn,
};
}
}

#include <libPShowdown/tries.tpp>

namespace LIBPSHOWDOWN_NAMESPACE {
namespace {
[[noreturn]] void unreachable(const char* message) {
	throw std::logic_error(std::string(message) + ": unreachable code reached");
}

inline const std::string EMPTY_STRING {};

inline const std::array<std::string, 18> TYPE_NAMES_INVERSE {"bug", "dragon", "electric", "fighting", "fire", "flying",
		"ghost", "grass", "ground", "ice", "normal", "poison", "psychic", "rock", "water", "dark", "steel", "fairy", };

using nature_subarray = std::array<std::string, 6>;

inline const std::array<nature_subarray, 6> NATURE_NAMES_INVERSE {nature_subarray {"hardy", "lonely", "adamant",
		"naughty", "brave"}, nature_subarray {"bold", "docile", "impish", "lax", "relaxed"}, nature_subarray {"modest",
		"mild", "bashful", "rash", "quiet"}, nature_subarray {"calm", "gentle", "careful", "quirky", "sassy"},
		nature_subarray {"timid", "hasty", "jolly", "naive", "serious"}, };

/**
 * Look up a single character in a trie, starting at the offset.
 * If the character is an ASCII letter, make it lowercase.
 *
 * If the character is 0 and it was found, set `value` to the
 * associated value, don't change the offset and return true.
 *
 * If the character is NOT 0 and it was found, don't change `value`,
 * set `offset` to the new offset and return true.
 *
 * If the character was not found, change nothing and return false.
 */
inline bool lookup(const uint_fast16_t* trie, uint_fast16_t& offset, char what, uint_fast16_t& value) {
	// ASCII case conversion
	if(what >= 'A' && what <= 'Z') {
		what += ('a' - 'A');
	}
	auto next_offset = offset;
	auto u16what = static_cast<uint_fast16_t>(what);
	while(true) {
		const uint_fast16_t* ptr_next {trie + next_offset};
		auto next = *ptr_next;
		if(u16what < next) {
			next_offset = *(ptr_next + 1);
		} else if(u16what > next) {
			next_offset = *(ptr_next + 2);
		} else if(!u16what) {
			value = *(ptr_next + 3);
			return true;
		} else {
			offset = *(ptr_next + 3);
			return true;
		}

		if(!next_offset) {
			return false;
		}
		// else continue our search
	}
}

/**
 * Look up a string in a trie and return the value. If not found, throw std::out_of_range.
 * Ensure the string is ASCII and automatically convert to lower case (if
 * not, throw std::out_of_range).
 * If ignore_punctuation is true, ignore ' ', '-' and '_'
 */
inline uint_fast16_t lookup(const uint_fast16_t* trie, const std::string& what, const std::string& caller,
		bool ignore_punctuation = false) {
	uint_fast16_t offset {0};
	uint_fast16_t value;

	auto xlookup = [&](char c) -> void {
		if(!lookup(trie, offset, c, value)) {
			throw std::out_of_range(caller + std::string(" - key not found: ") + what);
		}
	};

	for(char c : what) {
		if(c < ' ' || c > '~') {
			throw std::out_of_range(caller + std::string(" - key not found: ") + what);
		}
		if(ignore_punctuation && (c == ' ' || c == '-' || c == '_')) {
			continue;
		}
		xlookup(c);
	}
	xlookup(0);
	return value;
}
} // end of anonymous namespace

inline Type getType(const std::string& name) {
	return static_cast<Type>(lookup(TYPE_NAMES, name, "getType"));
}

inline const std::string& getNameOfType(Type type) {
	if(type == Type::NONE) {
		return EMPTY_STRING;
	} else if(type == Type::UNKNOWN || type == Type::NONSTANDARD) {
		return UNKNOWN;
	} else {
		return TYPE_NAMES_INVERSE.at(static_cast<size_t>(type) - 1);
	}
}

inline NVStatus getNVStatus(const std::string& name) {
	return static_cast<NVStatus>(lookup(NVSTATUS_NAMES, name, "getNVStatus"));
}

inline Nature getNature(const std::string& name) {
	return static_cast<Nature>(lookup(NATURES, name, "getNature"));
}

inline const std::string& getNameOfNature(Nature nature) {
	if(nature == Nature::NONE) {
		return EMPTY_STRING;
	} else if(nature == Nature::UNKNOWN) {
		return UNKNOWN;
	} else {
		return NATURE_NAMES_INVERSE.at((static_cast<size_t>(nature) >> 8) - 1).at(
				((static_cast<size_t>(nature) >> 4) & 0xF) - 1);
	}
}

namespace {
inline void assertSerializable(Nature nature, const std::string& info1, const std::string& info2) {
	if(static_cast<int>(nature) < 0x110) {
		throw std::invalid_argument(info1 + ": " + info2 + ": non-serializable nature" + getNameOfNature(nature));
	}
}

inline void assertSerializable(int value, const std::string& info1, const std::string& info2) {
	if(value < 0) {
		throw std::invalid_argument(
				info1 + ": " + info2 + ": invalid or non-serializable value " + std::to_string(value));
	}
}

inline void assertSerializable(const Stats& stats, const std::string& info1, const std::string& info2) {
	for(int i {0}; i <= Stats::MAX_INDEX; ++i) {
		if(stats.at(i) < 0) {
			throw std::invalid_argument(
					info1 + ": " + info2 + ": stat with field_index " + std::to_string(i)
							+ " has an unknown or non-serializable value " + std::to_string(stats.at(i)));
		}
	}
}

inline void assertSerializable(const std::string& value, const std::string& info1, const std::string& info2) {
	if(value == UNKNOWN) {
		throw std::invalid_argument(info1 + ": " + info2 + ": value is UNKNOWN, which it may not be");
	}
	for(char c : value) {
		if(c == '|') {
			throw std::invalid_argument(info1 + ": " + info2 + ": value contains '|', which it may not");
		}
	}
}

inline nlohmann::json statsToJSON(const Stats& stats) {
	nlohmann::json result { {"hp", stats.HP}, {"atk", stats.attack}, {"def", stats.defense},
			{"spa", stats.specialAttack}, {"spd", stats.specialDefense}, {"spe", stats.speed}, };
	return result;
}

inline nlohmann::json monsterToJSON(int generation, const Monster& monster) {
	const std::string error_message {"Serializing Monster as JSON"};

	assertSerializable(monster.EV, error_message, "EV");
	assertSerializable(monster.happiness, error_message, "happiness");
	assertSerializable(monster.IV, error_message, "IV");
	assertSerializable(monster.level, error_message, "level");
	assertSerializable(monster.nickname, error_message, "nickname");
	assertSerializable(monster.species, error_message, "species");

	nlohmann::json result { {"evs", statsToJSON(monster.EV)}, {"happiness", monster.happiness}, {"ivs", statsToJSON(
			monster.IV)}, {"level", monster.level}, {"name",
			monster.nickname.empty() ? monster.species : monster.nickname}, {"species", monster.species}, };

	if(generation >= 2) {
		assertSerializable(monster.ability, error_message, "ability");
		result["ability"] = monster.ability;
		if(!monster.item.empty()) {
			assertSerializable(monster.item, error_message, "item");
			result["item"] = monster.item;
		}
		switch(monster.gender) {
		case Gender::FEMALE:
			result["gender"] = "F";
			break;
		case Gender::MALE:
			result["gender"] = "M";
			break;
		case Gender::NONE:
			result["gender"] = "N";
			break;
		}
		if(generation >= 3) {
			assertSerializable(monster.nature, error_message, "nature");
			result["nature"] = getNameOfNature(monster.nature);
			if(!monster.ball.empty()) {
				assertSerializable(monster.ball, error_message, "ball");
				result["ball"] = monster.ball;
			}
		}
	}

	bool done_serializing {false};
	result["moves"] = nlohmann::json::array();
	for(size_t i {0}; i < monster.moves.size(); ++i) {
		const auto& move_id = monster.moves.at(i).ID;
		if(done_serializing) {
			if(!move_id.empty()) {
				throw std::invalid_argument("A Monster has moves after an empty slot");
			}
		} else if(move_id.empty()) {
			if(i == 0) {
				throw std::invalid_argument("A Monster has no move in its first move slot");
			} else {
				done_serializing = true;
			}
		} else {
			assertSerializable(move_id, error_message, monster.species);
			result["moves"].push_back(move_id);
		}
	}

	return result;
}
} // end of anonymous namespace

inline nlohmann::json teamToJSON(int generation, const std::vector<Monster>& team) {
	auto result = nlohmann::json::array();
	for(const auto& monster : team) {
		result.push_back(monsterToJSON(generation, monster));
	}
	return result;
}

inline Type calculateHiddenPowerType(int generation, const Stats& preHyperIV) {
	std::array<Type, 16> types {Type::FIGHTING, Type::FLYING, Type::POISON, Type::GROUND, Type::ROCK, Type::BUG,
			Type::GHOST, Type::STEEL, Type::FIRE, Type::WATER, Type::GRASS, Type::ELECTRIC, Type::PSYCHIC, Type::ICE,
			Type::DRAGON, Type::DARK, };

	if(generation < 2) {
		throw std::invalid_argument("Generation < 2");
	} else if(generation == 2) {
		return types.at(((preHyperIV.attack & 0b11) << 2) | (preHyperIV.defense & 0b11));
	} else {
		int x {(preHyperIV.HP & 1) | ((preHyperIV.attack & 1) << 1) | ((preHyperIV.defense & 1) << 2)
				| ((preHyperIV.speed & 1) << 3) | ((preHyperIV.specialAttack & 1) << 4)
				| ((preHyperIV.specialDefense & 1) << 5)};
		return types.at(x * 15 / 63);
	}
}

inline int calculateHiddenPowerPower(int generation, const Stats& preHyperIV) {
	if(generation < 2) {
		throw std::invalid_argument("Generation < 2");
	} else if(generation == 2) {
		int x {((preHyperIV.specialAttack / 2) >> 7) | (((preHyperIV.speed / 2) & 8) >> 6)
				| (((preHyperIV.defense / 2) & 8) >> 5) | (((preHyperIV.attack / 2) & 8) >> 4)};
		return (5 * x + (preHyperIV.specialAttack / 2) % 4) / 2 + 31;
	} else {
		int x {((preHyperIV.HP & 2) >> 1) | ((preHyperIV.attack & 2)) | ((preHyperIV.defense & 2) << 1)
				| ((preHyperIV.speed & 2) << 3) | ((preHyperIV.specialAttack & 2) << 4)
				| ((preHyperIV.specialDefense & 2) << 5)};
		return x * 40 / 63 + 30;
	}
}

inline Stats::Stats(bool) noexcept :
		Stats(-1, -1, -1, -1, -1, -1) {
}

inline Stats::Stats(int hp, int attack, int defense, int special_attack, int special_defense, int speed) noexcept :
		HP(hp), attack(attack), defense(defense), specialAttack(special_attack), specialDefense(special_defense), speed(
				speed) {
}

inline int Stats::indexOf(const std::string& field_name) {
	return static_cast<int>(lookup(STAT_NAMES, field_name, "Stats::indexOf"));
}

inline int& Stats::at(int field_index) {
	switch(field_index) {
	case 0:
		return HP;
	case 1:
		return attack;
	case 2:
		return defense;
	case 3:
		return specialAttack;
	case 4:
		return specialDefense;
	case 5:
		return speed;
	default:
		throw std::invalid_argument(std::string("Invalid Stats field index: ") + std::to_string(field_index));
	}
}

inline const int& Stats::at(int field_index) const {
	switch(field_index) {
	case 0:
		return HP;
	case 1:
		return attack;
	case 2:
		return defense;
	case 3:
		return specialAttack;
	case 4:
		return specialDefense;
	case 5:
		return speed;
	default:
		throw std::invalid_argument(std::string("Invalid Stats field index: ") + std::to_string(field_index));
	}
}

inline int& Stats::at(const std::string& field_name) {
	return at(indexOf(field_name));
}

inline const int& Stats::at(const std::string& field_name) const {
	return at(indexOf(field_name));
}

inline MoveSlot::MoveSlot(bool) :
		MoveSlot(UNKNOWN, -1, -1, UNKNOWN) {
}

inline MoveSlot::MoveSlot(const std::string& ID, int PP, int maxPP, const std::string& name) :
		PP(PP), maxPP(maxPP), ID(ID), name(name) {
}

inline MoveSlot::operator bool() noexcept {
	return !this->ID.empty();
}

namespace {
/// Find the next character from `chars` and return a pointer to it
/// Return nullptr if not found
template<unsigned int N>
const char* findChar(const char* start, const char* end, std::array<char, N> chars) {
	while(start != end) {
		char c = *start;
		for(char target : chars) {
			if(target == c) {
				return start;
			}
		}
		++start;
	}
	// If not found
	return nullptr;
}

/// Translate a position as reported by Showdown into what we use
int translatePosition(const std::string& position, BattleState::Category category, int our_id) {
	const uint_fast16_t* trie;

	switch(category) {
	case BattleState::Category::SINGLES:
		if(our_id == 1) {
			return (position == "p1a" ? 1 : -1);
		} else if(our_id == 2) {
			return (position == "p2a" ? 1 : -1);
		} else {
			unreachable("translatePosition, SINGLES");
		}
	case BattleState::Category::FREE_FOR_ALL:
		throw std::runtime_error("Sorry, free for all position lookups not implemented yet");
	case BattleState::Category::DOUBLES:
		if(our_id == 1) {
			trie = TRANSLATE_POSITION_DOUBLES_P1;
		} else if(our_id == 2) {
			trie = TRANSLATE_POSITION_DOUBLES_P2;
		} else {
			unreachable("translatePosition, DOUBLES");
		}
		break;
	case BattleState::Category::TRIPLES:
		if(our_id == 1) {
			trie = TRANSLATE_POSITION_TRIPLES_P1;
		} else if(our_id == 2) {
			trie = TRANSLATE_POSITION_TRIPLES_P2;
		} else {
			unreachable("translatePosition, TRIPLES");
		}
		break;
	case BattleState::Category::MULTI:
		switch(our_id) {
		case 1:
			trie = TRANSLATE_POSITION_MULTI_P1;
			break;
		case 2:
			trie = TRANSLATE_POSITION_MULTI_P2;
			break;
		case 3:
			trie = TRANSLATE_POSITION_MULTI_P3;
			break;
		case 4:
			trie = TRANSLATE_POSITION_MULTI_P4;
			break;
		default:
			unreachable("translatePosition, MULTI");
		}
	}

	try {
		auto result = lookup(trie, position, "translatePosition", false);
		return static_cast<int>(result) - 4;
	} catch(std::out_of_range&) {
		throw std::runtime_error(std::string("Invalid position: ") + position);
	}
}

/// Abstract base for DirectSimConnection, AuxSimConnection and ServerConnection
/// The parser is implemented in a less efficient, but also less error-prone way
/// (TODO: maybe write a more efficient implementation one day)
class ServerReplyProcessor {
public:
	virtual ~ServerReplyProcessor() = default;
protected:
	/// Called when all basic data has been collected and the game is starting
	/// (before team preview)
	virtual void initializeBattle(BattleState::Category category, int generation) = 0;

	/**
	 * Called when some info about a player has been received.
	 *
	 * @param    player_id    ID of the player from 1 to 4
	 * @param    name         Name of the player as reported by Showdown
	 * @param    avatar       A string denoting the player's avatar, should probably be ignored
	 *     when writing a bot
	 * @param    rating       The player's ELO rating, 0 if unknown
	 *
	 * @return If not in a single-player mode (e.g. DirectSimConnection), always return false.
	 *     In single-player mode return `true` if this is the player we're supposed to be and
	 *     `false` for all others.
	 */
	virtual bool setPlayerInfo(int player_id, std::string name, std::string avatar, int rating);

	/**
	 * Called when we need the battle state for the given player.
	 *
	 * @param    player_id    1-based player ID. If running in single player mode,
	 *     the ID will be 1.
	 */
	virtual BattleState& getBattleState(int playe_idr) = 0;

	/// Feed data into the parser
	void feed(const char* start, const char* end);
private:
	// Types
	enum class TeamPreviewStatus {
		NOT_STARTED,
		RECEIVING,
		FINISHED,
	};

	// Basic parser fields
	void (ServerReplyProcessor::*parserCurrent)();
	void (ServerReplyProcessor::*parserCallback)(bool);
	const char* current, * end;
	ServerReply currentReply {ServerReply::EMPTY_LINE};

	// Game state
	bool gameTrulyStarted {false};
	bool observerMode {false};
	BattleState::Category battleCategory {BattleState::Category::SINGLES};
	TeamPreviewStatus TPS;
	int maxPlayerID {0};
	int ourPlayerID {0};   // Only in single player mode
	unsigned int generation {0};

	// Buffers used during a single reply
	std::string buffer {};
	std::string bufferedString {};
	bool bufferedBool {false};
	MonsterDetails bufferedMonsterDetails {};

	// Cross-reply buffers
	std::array<std::string, 4> bufferedRequests {};
	int currentPlayerID {1};	// Used when a reply has an associated player ID sent as another reply
	// 0 = no split, > 0 = currently split for this player, < 0 = previously split for this player
	int splitPlayerID {0};

	// Methods
	void parserEntryPoint();

	void readField();
	void skipTillNewline();
	void readRestOfLine();
	void assertEOL(bool eol);
	void setNextCallback(bool eol, void (ServerReplyProcessor::*callback)(bool));

	void parseReplyHeader(bool eol);

	// Pre-start
	// Zero-argument replies
	void process_playerID(int player);
	void process_sideupdate();
	void process_start();
	void process_update();

	// One-argument replies
	void parse_gametype(bool eol);
	void parse_gen(bool eol);
	void parse_rule(bool eol);
	void parse_split(bool eol);

	// Entire string
	void parse_request(bool eol);

	// Post-start
	// One-argument replies
	void parse_turn(bool eol);

	// Three-argument replies
	void parse_dragPosition(bool eol);
	void parse_switchPosition(bool eol);
	void parse_switchDragPosition(bool eol);
	void parse_switchDragDetails(bool eol);
	void parse_switchDragHP(bool eol);
};

inline void ServerReplyProcessor::feed(const char* start, const char* end) {
	current = start;
	this->end = end;

	while(current != this->end) {
		if(!parserCurrent) {
			parserEntryPoint();
		} else {
			(this->*parserCurrent)();
		}
	}
}

inline void ServerReplyProcessor::readField() {
	const char* fieldsep = findChar<3>(current, end, {'|', '\n'});
	if(!fieldsep) {
		buffer.append(current, end);
		current = end;
	} else {
		buffer.append(current, fieldsep);
		current = fieldsep + 1;
		parserCurrent = nullptr;
		(this->*parserCallback)(*fieldsep == '\n');
		buffer.clear();
	}
}

inline void ServerReplyProcessor::skipTillNewline() {
	while(current != end) {
		if(*current++ == '\n') {
			parserCurrent = nullptr;
			return;
		}
	}
}

inline void ServerReplyProcessor::readRestOfLine() {
	auto line_start = current;
	while(current != end) {
		if(*current++ == '\n') {
			parserCurrent = nullptr;
			buffer.append(line_start, current - 1);
			(this->*parserCallback)(true);
			buffer.clear();
			return;
		}
	}
	// If we reached the end without the line ending
	buffer.append(line_start, end);
}

inline void ServerReplyProcessor::setNextCallback(bool eol, void (ServerReplyProcessor::*callback)(bool)) {
	if(eol) {
		throw std::runtime_error("Unexpected end of line");
	} else {
		parserCallback = callback;
		parserCurrent = &ServerReplyProcessor::readField;
	}
}

inline void ServerReplyProcessor::parserEntryPoint() {
	if(*current == '|') {
		++current;
	}
	// TODO: process '>'
	parserCallback = &ServerReplyProcessor::parseReplyHeader;
	parserCurrent = &ServerReplyProcessor::readField;
}

inline void ServerReplyProcessor::parseReplyHeader(bool eol) {
	if(buffer.empty()) {
		if(eol) {
			// The server sent us an empty string, ignore it
			return;
		} else {
			// This is supposed to be processed as a chat message or something
			throw std::logic_error("Replies without header can't be processed yet");
		}
	}

	ServerReply rh;
	try {
		rh = static_cast<ServerReply>(lookup(SERVER_REPLY_HEADERS, buffer, "ServerReplyProcessor::parseReplyHeader"));
	} catch(std::out_of_range& e) {
		throw std::runtime_error(std::string("Unknown reply header: ") + buffer);
	}

	switch(rh) {
	case ServerReply::p1:
		process_playerID(1);
		assertEOL(eol);
		break;
	case ServerReply::p2:
		process_playerID(2);
		assertEOL(eol);
		break;
	case ServerReply::p3:
		process_playerID(3);
		assertEOL(eol);
		break;
	case ServerReply::p4:
		process_playerID(4);
		assertEOL(eol);
		break;
	case ServerReply::sideupdate:
		process_sideupdate();
		assertEOL(eol);
		break;
	case ServerReply::update:
		process_update();
		assertEOL(eol);
		break;
	case ServerReply::start:
		process_start();
		break;
	case ServerReply::gametype:
		setNextCallback(eol, &ServerReplyProcessor::parse_gametype);
		break;
	case ServerReply::gen:
		setNextCallback(eol, &ServerReplyProcessor::parse_gen);
		break;
	case ServerReply::request:
		setNextCallback(eol, &ServerReplyProcessor::parse_request);
		break;
	case ServerReply::rule:
		setNextCallback(eol, &ServerReplyProcessor::parse_rule);
		break;
	case ServerReply::split:
		setNextCallback(eol, &ServerReplyProcessor::parse_split);
		break;
	// Team preview
	case ServerReply::clearpoke:
	case ServerReply::poke:
	case ServerReply::teampreview:
		throw std::runtime_error("Team preview not implemented yet");
	// Post-start
	case ServerReply::drag:
		setNextCallback(eol, &ServerReplyProcessor::parse_dragPosition);
		break;
	case ServerReply::switch_position:
		setNextCallback(eol, &ServerReplyProcessor::parse_switchPosition);
		break;
	case ServerReply::turn:
		setNextCallback(eol, &ServerReplyProcessor::parse_turn);
		break;
	default:
		throw std::logic_error(
				std::string("Processing this server reply not implemented yet: ")
						+ std::to_string(static_cast<int>(rh)));
	}
}

inline void ServerReplyProcessor::process_sideupdate() {
	if(currentReply != ServerReply::EMPTY_LINE) {
		throw std::runtime_error(
				std::string("Server sent sideupdate while we were in ")
						+ std::to_string(static_cast<int>(currentReply)));
	}
	currentReply = ServerReply::sideupdate;
}

inline void ServerReplyProcessor::process_playerID(int player) {
	if(currentReply != ServerReply::sideupdate) {
		throw std::runtime_error(
				std::string("Server sent a player ID while we were in ")
						+ std::to_string(static_cast<int>(currentReply)));
	}
	currentPlayerID = player;
}

inline void ServerReplyProcessor::process_update() {
	// For now it seems we don't need it at all
	return;
}

inline void ServerReplyProcessor::parse_gametype(bool eol) {
	try {
		battleCategory = BattleState::categoryByName(buffer);
		if(battleCategory == BattleState::Category::MULTI
				|| battleCategory == BattleState::Category::FREE_FOR_ALL) {
			maxPlayerID = 4;
		} else {
			maxPlayerID = 2;
		}
	} catch(std::out_of_range&) {
		throw std::runtime_error("Unknown gametype");
	}
	assertEOL(eol);
}

inline void ServerReplyProcessor::parse_gen(bool eol) {
	try {
		int generation_pre = std::stoi(buffer);
		if(generation_pre < static_cast<int>(GENERATION_MIN) || generation_pre > static_cast<int>(GENERATION_MAX)) {
			throw std::out_of_range(buffer);
		}
		generation = static_cast<unsigned int>(generation_pre);
	}
	catch(std::invalid_argument&) {
		throw std::runtime_error(std::string("Invalid generation number: ") + buffer);
	}
	catch(std::out_of_range&) {
		throw std::runtime_error(std::string("Generation number out of range: ") + buffer);
	}
	assertEOL(eol);
}

inline void ServerReplyProcessor::parse_rule(bool eol) {
	// FIXME: implement
	assertEOL(eol);
}

inline void ServerReplyProcessor::parse_split(bool eol) {
	if(ourPlayerID) {
		throw std::runtime_error("split received while in single-player mode");
	}
	try {
		auto x = lookup(PLAYER_IDS, buffer, "ServerReplyProcessor::parse_split");
		splitPlayerID = static_cast<int>(x);
		assertEOL(eol);
	}
	catch(std::out_of_range&) {
		throw std::runtime_error(std::string("Invalid player ID: ") + buffer);
	}
}

inline void ServerReplyProcessor::parse_request(bool eol) {
	if(buffer.empty()) {
		throw std::runtime_error("Empty request");
	}
	// We either have one player or always have currentPlayerID set at this point
	if(gameTrulyStarted) {
		getBattleState(currentPlayerID)._registerRequest(buffer);
	} else {
		bufferedRequests.at(currentPlayerID - 1) = buffer;
	}
	assertEOL(eol);
}

inline void ServerReplyProcessor::parse_switchDragPosition(bool eol) {
	bufferedString = buffer;
	setNextCallback(eol, &ServerReplyProcessor::parse_switchDragDetails);
}

inline void ServerReplyProcessor::parse_switchPosition(bool eol) {
	bufferedBool = false;
	parse_switchPosition(eol);
}

inline void ServerReplyProcessor::parse_dragPosition(bool eol) {
	bufferedBool = true;
	parse_switchPosition(eol);
}

inline void ServerReplyProcessor::parse_switchDragDetails(bool eol) {
	bufferedMonsterDetails = MonsterDetails::fromString(buffer);
	setNextCallback(eol, &ServerReplyProcessor::parse_switchDragHP);
}

inline void ServerReplyProcessor::parse_switchDragHP(bool eol) {
	if(splitPlayerID > 0) {
		int position = translatePosition(bufferedString, battleCategory, splitPlayerID);
		getBattleState(splitPlayerID)._registerSwitch(bufferedBool, position, bufferedMonsterDetails,
				MonsterHP::fromString(bufferedString, true));
	} else if(splitPlayerID < 0) {
		for(int i {1}; i <= maxPlayerID; ++i) {
			if(i != -splitPlayerID) {
				int position = translatePosition(bufferedString, battleCategory, i);
				getBattleState(i)._registerSwitch(bufferedBool, position, bufferedMonsterDetails,
						MonsterHP::fromString(bufferedString, false));
			}
		}
		splitPlayerID = 0;
	} else if(ourPlayerID) {
		int position = translatePosition(bufferedString, battleCategory, ourPlayerID);
		bool hp_exact = (position == 1) || (position >= 2 && battleCategory != BattleState::Category::MULTI);
		getBattleState(ourPlayerID)._registerSwitch(bufferedBool, position, bufferedMonsterDetails,
				MonsterHP::fromString(bufferedString, hp_exact && !observerMode));
	} else {
		throw std::runtime_error("Received a switch without a split while not in single-player mode");
	}
	assertEOL(eol);
}
} // end of anonymous namespace
}
