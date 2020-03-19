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
#define SEIMEI_LIBPSHOWDOWN_HPP_

//@formatter:off

#if !defined(LIBPSHOWDOWN_POSIX) && !defined(LIBPSHOWDOWN_WINDOWS) && !defined(LIBPSHOWDOWN_NO_DIRECTSIM)
#error Either LIBPSHOWDOWN_POSIX, or LIBPSHOWDOWN_WINDOWS, or LIBPSHOWDOWN_NO_DIRECTSIM must be defined
#elif (defined(LIBPSHOWDOWN_POSIX) + defined(LIBPSHOWDOWN_WINDOWS) + defined(LIBPSHOWDOWN_NO_DIRECTSIM)) > 1
#error Only one of LIBPSHOWDOWN_POSIX, LIBPSHOWDOWN_WINDOWS and LIBPSHOWDOWN_NO_DIRECTSIM must be defined
#endif

#ifdef LIBPSHOWDOWN_WINDOWS
#error Windows support not implemented yet
#endif

#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#ifndef LIBPSHOWDOWN_NAMESPACE
#define LIBPSHOWDOWN_NAMESPACE pshowdown
#endif

namespace LIBPSHOWDOWN_NAMESPACE {
inline const unsigned int GENERATION_MIN {1};
inline const unsigned int GENERATION_MAX {8};

inline const std::string UNKNOWN {"?"};

inline const unsigned long
	/// 2 Ability Clause: Limit two of each ability [appears in Hackmons]
	RULE_2_ABILITY_CLAUSE					{1ul},
	/// 3 Baton Pass Clause: Limit three Baton Passers
	RULE_3_BATON_PASS_CLAUSE				{1ul << 1},
	/// Accuracy Moves Clause: Accuracy-lowering moves are banned
	RULE_ACCURACY_MOVES_CLAUSE				{1ul << 2},
	/// Baton Pass Clause: Limit one Baton Passer, can't pass Spe and other stats simultaneously
	RULE_BATON_PASS_CLAUSE					{1ul << 3},
	/// CFZ Clause: Crystal-free Z-Moves are banned [appears in hHackmons]
	RULE_CFZ_CLAUSE							{1ul << 4},
	/// Dynamax Clause: You cannot dynamax
	RULE_DYNAMAX_CLAUSE						{1ul << 5},
	/// Endless Battle Clause: Forcing endless battles is banned
	RULE_ENDLESS_BATTLE_CLAUSE				{1ul << 6},
	/// Evasion Abilities Clause: Evasion abilities are banned
	RULE_EVASION_ABILITIES_CLAUSE			{1ul << 7},
	/// Evasion Moves Clause: Evasion moves are banned
	RULE_EVASION_MOVES_CLAUSE				{1ul << 8},
	/// Exact HP Mod: Exact HP is shown
	RULE_EXACT_HP_MOD						{1ul << 9},
	/// Freeze Clause Mod: Limit one foe frozen [Showdown will enforce it server-side]
	RULE_FREEZE_CLAUSE_MOD					{1ul << 10},
	/// HP Percentage Mod: HP is shown in percentages [as opposed to /48; the library processes it automatically]
	RULE_HP_PERCENTAGE_MOD					{1ul << 11},
	/// Inverse Mod: Weaknesses become resistances, while resistances and immunities become weaknesses.
	RULE_INVERSE_MOD						{1ul << 12},
	/// Item Clause: Limit one of each item
	RULE_ITEM_CLAUSE						{1ul << 13},
	/// Mega Rayquaza Clause: You cannot mega evolve Rayquaza
	RULE_MEGA_RAYQUAZA_CLAUSE				{1ul << 14},
	/// Moody Clause: Moody is banned
	RULE_MOODY_CLAUSE						{1ul << 15},
	/// NFE Clause: Fully Evolved Pokémon are banned
	RULE_NFE_CLAUSE							{1ul << 16},
	/// OHKO Clause: OHKO moves are banned
	RULE_OHKO_CLAUSE						{1ul << 17},
	/// Same Type Clause: Pokémon in a team must share a type
	RULE_SAME_TYPE_CLAUSE					{1ul << 18},
	/** @brief Sleep Clause Mod: Limit one foe put to sleep
	 *
	 * It also prevents Mega-Gengar from knowing Hypnosis (on the basis that
	 * Hypnosis + Shadow Tag are worse than unrestricted sleep).
	 *
	 * This is (at the moment of writing) the only clause which Showdown enforces
	 * by deviating from cartridge mechanics: if a foe is put to sleep in violation
	 * of the clause, Showdown cancels the sleep status.
	 */
	RULE_SLEEP_CLAUSE_MOD					{1ul << 19},
	/// Species Clause: Limit one of each Pokémon
	RULE_SPECIES_CLAUSE						{1ul << 20},
	/// Swagger Clause: Swagger is banned
	RULE_SWAGGER_CLAUSE						{1ul << 21},
	/// Switch Priority Clause Mod: Faster Pokémon switch first
	RULE_SWITCH_PRIORITY_CLAUSE_MOD			{1ul << 22},
	/// Z-Move Clause: Z-Moves are banned
	RULE_ZMOVE_CLAUSE						{1ul << 23};


/**
 * A Pokémon or Move type
 */
enum class Type {
	NONE = 0,
	UNKNOWN = 100,
	/// May only occur in non-standard formats
	NONSTANDARD = 200,

	BUG = 1,
	DRAGON = 2,
	ELECTRIC = 3,
	FIGHTING = 4,
	FIRE = 5,
	FLYING = 6,
	GHOST = 7,
	GRASS = 8,
	GROUND = 9,
	ICE = 10,
	NORMAL = 11,
	POISON = 12,
	PSYCHIC = 13,
	ROCK = 14,
	WATER = 15,
	// Generation 2+
	DARK = 16,
	STEEL = 17,
	// Generation 6+
	FAIRY = 18,
};

/**
 * Get a Type value by the name of the type. "NONE", "UNKNOWN"
 * and "NONSTANDARD" are not valid type names.
 *
 * @param    name    Case-insensitive type name, e.g. "ELECTRIC",
 *     or "fighting", or "Fairy"
 *
 * @return The corresponding Type value.
 * @throw std::out_of_range if ``name`` is not a valid type name.
 */
Type getType(const std::string& name);

/**
 * Get the name of Type as a lowercase string, e.g. "electric" for Type::ELECTRIC
 * The string will be allocated statically. Type::NONE returns an empty string,
 * Type::UNKNOWN and Type::NONSTANDARD will return @ref UNKNOWN
 */
const std::string& getNameOfType(Type type);

/// Pokémon gender
enum class Gender {
	UNKNOWN,
	NONE,
	FEMALE,
	MALE,
};

/// Non-volatile status conditions
enum class NVStatus {
	NONE,
	/// May only occur in non-standard formats
	NONSTANDARD,

	/// aka "FNT"
	FAINTED,
	/// aka "BRN"
	BURN,
	/// aka "FRZ" or "FROZEN"
	FREEZE,
	/// aka "PAR"
	PARALYSIS,
	/// aka "PSN"
	POISON,
	/// aka "TOX"; the type of poison instilled by the move "Toxic" etc.
	TOXIC,
	/// aka "SLP"
	SLEEP,
};

/**
 * Get an NVStatus value by the name of the status. "NONE" and "NONSTANDARD"
 * are not valid names, but an empty string will return NVStatus::NONE.
 *
 * @param    name    Case-insensitive name or abbreviation of the status,
 *     e.g. "Paralysis" or "par"
 *
 * @return The corresponding NVStatus value.
 * @throw std::out_of_range if ``name`` is not a valid name of a status.
 */
NVStatus getNVStatus(const std::string& name);

/// A Pokémon's nature
enum class Nature {
	NONE = 0,
	UNKNOWN = 1,
	// The values are such that (value >> 8) == increased stat, ((value >> 4) & 0xF) == decreased stat
	// + Attack, likes Spicy foods
	HARDY = 0x110, LONELY = 0x120, ADAMANT = 0x130, NAUGHTY = 0x140, BRAVE = 0x150,
	// + Defense, likes Sour foods
	BOLD = 0x210, DOCILE = 0x220, IMPISH = 0x230, LAX = 0x240, RELAXED = 0x250,
	// + Special Attack, likes Dry foods
	MODEST = 0x310, MILD = 0x320, BASHFUL = 0x330, RASH = 0x340, QUIET = 0x350,
	// + Special Defense, likes Bitter foods
	CALM = 0x410, GENTLE = 0x420, CAREFUL = 0x430, QUIRKY = 0x440, SASSY = 0x450,
	// + Speed, likes Sweet foods
	TIMID = 0x510, HASTY = 0x520, JOLLY = 0x530, NAIVE = 0x540, SERIOUS = 0x550,
};

/**
 * Get a Nature by its name. "NONE" and "UNKNOWN" are not valid names.
 *
 * @param    name    Case-insensitive name of the nature, e.g. "Hardy".
 *
 * @return The corresponding Nature value.
 * @throw std::out_of_range if ``name`` is not a valid name of a nature.
 */
Nature getNature(const std::string& name);

/**
 * Get the name of a Nature as a lowercase string, e.g. "hardy" for Nature::HARDY.
 * The string will be allocated statically. Nature:NONE will return an empty string,
 * while Nature::UNKNOWN will return @ref UNKNOWN.
 */
const std::string& getNameOfNature(Nature nature);

/**
 * Move target specification
 *
 * Note that the values follow a bit pattern (high to low):
 *
 * 1. Does it need a target?
 * 2. Indicates a special value which may deviate from the usual rules.
 *      If bit 1 is set, bit 2 will never be set.
 * 3. Can target self?
 * 4. Can target adjacent allies?
 * 5. Can target adjacent foes?
 * 6. Can target non-adjacent allies?
 * 7. Can target non-adjacent foes?
 *
 * _BITSET_* values can be used by users as well.
 */
enum class MoveTarget {
	_BITSET_TARGETED        = 0b1000000,
	_BITSET_SPECIAL         = 0b0100000,
	_BITSET_SELF            = 0b0010000,
	_BITSET_ADJACENT_ALLIES = 0b0001000,
	_BITSET_ADJACENT_FOES   = 0b0000100,
	_BITSET_FAR_ALLIES      = 0b0000010,
	_BITSET_FAR_FOES        = 0b0000001,

	/**
	 * Placeholder for when the target is not actually set
	 */
	UNKNOWN = 0,

	/**
	 * The move selects its own target according to some algorithm.
	 */
	SCRIPTED = 0b0111111,

	/*
	 * The move affects the Pokémon that used it
	 */
	SELF = _BITSET_SELF,

	/**
	 * The move affects an adjacent foe at random, possibly a
	 * different foe each turn
	 *
	 * Alias: "randomNormal"
	 */
	RANDOM_ADJACENT_FOE = _BITSET_SPECIAL | _BITSET_ADJACENT_FOES,

	/**
	 * The move affects all Pokémon on the allied side
	 */
	ALLIED_SIDE = _BITSET_SELF | _BITSET_ADJACENT_ALLIES | _BITSET_FAR_ALLIES,

	/**
	 * The move targets all Pokémon on the allied *team*, even
	 * those not on the field, but not the ally player in
	 * multi-battles
	 *
	 * (TODO: verify? Set _BITSET_ADJAENT_ALLIES and _BITSET_FAR_ALLIES
	 * if it actually does also target the allied player)
	 */
	ALLIED_TEAM = _BITSET_SPECIAL | _BITSET_SELF,

	/**
	 * The move affects all Pokémon on the foe side
	 */
	FOE_SIDE = _BITSET_ADJACENT_FOES | _BITSET_FAR_FOES,

	/**
	 * The move affects all adjacent Pokémon
	 */
	ALL_ADJACENT = _BITSET_ADJACENT_FOES | _BITSET_ADJACENT_ALLIES,

	/**
	 * The move affects all adjacent foes
	 */
	ALL_ADJACENT_FOES = _BITSET_ADJACENT_FOES,

	/**
	 * The move affects all Pokémon on the field
	 */
	ALL = 0b0011111,

	/**
	 * Targeted move: any adjacent Pokémon
	 *
	 * Alias: "normal"
	 */
	ADJACENT = _BITSET_TARGETED | _BITSET_ADJACENT_ALLIES | _BITSET_ADJACENT_FOES,

	/**
	 * Targeted move: one adjacent ally
	 */
	ADJACENT_ALLY = _BITSET_TARGETED | _BITSET_ADJACENT_ALLIES,

	/**
	 * Targeted move: one adjacent ally or self
	 */
	ADJACENT_ALLY_OR_SELF = _BITSET_TARGETED | _BITSET_SELF | _BITSET_ADJACENT_ALLIES,

	/**
	 * Targeted move: one adjacent foe
	 */
	ADJACENT_FOE = _BITSET_TARGETED | _BITSET_ADJACENT_FOES,

	/**
	 * Targeted move: any one Pokémon on the field EXCEPT self
	 */
	ANY = 0b1001111,
};

/// Different ways a move can be used
enum class MoveModifier {
	/// Normally
	NONE,
	/// Mega-Evolve (or use Primal Reversion or Ultra Burst), then use the move
	MEGA,
	/// As a Z-Move
	Z,
	/// Dynamax or Gigantamax, then use the move
	DYNAMAX,
};


/// Structure to represent Pokémon stats and stat-related data
struct Stats {
	/// Hit points, aka "hp", index 0
	int HP {0};
	/// (Physical) Attack, aka "atk", index 1
	int attack {0};
	/// (Physical) Defense, aka "def", index 2
	int defense {0};
	/// Special Attack, aka "spa", index 3
	int specialAttack {0};
	/**
	 * Special Defense, aka "spd", index 4.
	 * In Generation I always equals specialAttack.
	 */
	int specialDefense {0};
	/// Speed, aka "spe", index 5
	int speed {-1};

	static const int MAX_INDEX {5};

	/*
	 * Get the index of a field given its name.
	 *
	 * @param    field_name    Name of the field. Can be the full name
	 *     (e.g. "SpecialAttack" or "HP"), or the abbreviation
	 *     noted in the field comment, like "spa". It's case-insensitive;
	 *     spaces, underscores and hyphen-minus (chars ' ', '_' and '-')
	 *     are ignored.
	 *
	 * @return index of the field, as noted in the field comment
	 * @throw std::out_of_range if the string is not a valid field name
	 *     (in keeping with how containers usually work)
	 */
	static int indexOf(const std::string& field_name);

	/// Construct a structure filled with zeros
	Stats() noexcept = default;

	/**
	 * Construct a structure filled with -1 (all stats unknown).
	 * The value of the argument will be ignored.
	 */
	Stats(bool) noexcept;

	/// Construct a structure filled with provided values
	Stats(int hp, int attack, int defense, int special_attack, int special_defense, int speed) noexcept;

	/**
	 * Get reference to a field by its index.
	 *
	 * @throw std::out_of_range if the index is invalid
	 */
	int& at(int field_index);
	const int& at(int field_index) const;

	/**
	 * Get reference to a field by its name or abbreviation. See
	 * @ref indexOf for more info on field names and abbreviations.
	 *
	 * @throw std::out_of_range if the index is invalid
	 */
	int& at(const std::string& field_name);
	const int& at(const std::string& field_name) const;
};

Type calculateHiddenPowerType(int generation, const Stats& preHyperIV);
int calculateHiddenPowerPower(int generation, const Stats& preHyperIV);


/// Representation of a move on a Pokémon
struct MoveSlot {
	/// If true, the move has been disabled
	bool disabled {false};

	/// Current number of Power Points (remaining uses)
	int PP {0};

	/**
	 * Starting number of Power Points.
	 *
	 * The library will never pass MoveSlots with PP > maxPP, but
	 * if you need to experiment, you may.
	 */
	int maxPP {0};

	/// Acceptable targets of the move
	MoveTarget target {MoveTarget::UNKNOWN};

	/// ID of the move in Showdown, like "voltabsorb"
	std::string ID {};

	/**
	 * Name of the move, like "Volt Absorb"
	 *
	 * When the name is known, the ID is also always known, but the name may
	 * be UNKNOWN while the ID is actually available, because Showdown mostly
	 * uses IDs to identify moves.
	 */
	std::string name {};


	/// Construct a MoveSlot with no ID and no PP, representing no move
	MoveSlot() noexcept = default;

	/**
	 * Construct a MoveSlot with the ID equal to @ref UNKNOWN,
	 * PP = -1 and maxPP = -1. The value of the argument will be ignored.
	 */
	MoveSlot(bool);

	/// Construct according to the arguments.
	MoveSlot(const std::string& ID, int PP, int maxPP, const std::string& name = {});

	/// Return true if the move has an ID (even UNKNOWN), false otherwise
	operator bool() noexcept;
};


/// This class can be subclassed by uses to add arbitrary data to various
/// data structures.
class UserData {
public:
	virtual ~UserData();
};


/// Representation of a Pokémon in battle
struct Monster {
	/**
	 * The Pokémon is shiny. Flair-only, should have no effect on
	 * the actual gameplay.
	 *
	 * If the shininess is unknown, it will be presumed false.
	 */
	bool shiny {false};

	/**
	 * Gender of the Pokémon. Gender::UNKNOWN if unknown, always
	 * Gender::NONE in Generation I. Otherwise Gender::FEMALE,
	 * Gender::MALE or, if the Pokémon has no gender, Gender::NONE.
	 */
	Gender gender {Gender::UNKNOWN};

	/**
	 * Pokémon's nature. Nature::UNKNOWN if unknown, Nature::NONE in
	 * Generation I.
	 */
	Nature nature {Nature::UNKNOWN};

	/**
	 * Pokémon's current status condition. It cannot be unknown: Pokémon
	 * that haven't been sent out will have no status condition.
	 */
	NVStatus status {NVStatus::NONE};

	/**
	 * Pokémon's level (from 1 to 100), -1 if unknown
	 */
	int level {-1};

	/**
	 * Pokémon's happiness (from 0 to 255), -1 if unknown
	 */
	int happiness {-1};

	/**
	 * This will be -1 unless .status == NVStatus.TOXIC. In the latter case
	 * it will contain the number of times the Pokémon has been damaged
	 * by the toxic poison, i.e. 1 after the same turn ends, 2 after the
	 * end of the next turn and so forth. It resets to 0 when the Pokémon
	 * is switched out, unless the poison is cured (in which case it is,
	 * again, set to -1).
	 */
	int toxicTurns {-1};

	/**
	 * The Pokémon's current exact HP. If unknown (e.g. because it's not
	 * your Pokémon), will be set to -1.
	 */
	int HP {-1};

	/**
	 * The Pokémon's maximum HP, which may also change during the battle
	 * (e.g. due to Dynamax). If unknown (e.g. because it's not
	 * your Pokémon), will be set to -1.
	 */
	int maxHP {-1};

	/// Accuracy boost, from -6 to 6
	int accuracyBoost {0};

	/// Evasion boost, from -6 to 6
	int evasionBoost {0};

	/**
	 * natureIncreases and @ref natureDecreases are values from 0 to
	 * @ref Stats::MAX_INDEX and they are the indices (see @ref Stats::at)
	 * of stats that are, respectively, increased and decreased by the
	 * Pokémon's nature. If `natureIncreases == natureDecreases`, the Pokémon's
	 * stats are not changed by its nature.
	 *
	 * The reason why it is resolved automatically rather than requiring
	 * the caller to do it on its own is that it's very simple, and in
	 * Generation VIII natures are no longer explicitly named in-game.
	 *
	 * Both fields will equal -1 if .nature is NONE or UNKNOWN.
	 */
	int natureIncreases {-1};
	int natureDecreases {-1};

	/**
	 * The position the Pokémon is currently occupying: -3, -2, -1,
	 * 1, 2, 3 or 0 if it's not on the field. See @ref BattleState
	 * for more information
	 */
	int position {0};

	/**
	 * 1-based index of the Pokémon in its team or 0 if it is an
	 * abstract Pokémon that isn't on any team. See @ref BattleState
	 * for more information
	 */
	unsigned int teamIndex {0};

	/**
	 * The Pokémon's remaining HP fraction (from 0.0f to 1.0f). Your own
	 * Pokémon's actual HP will be in .HP, but the HP of opposing
	 * Pokémon is only visible as this fraction.
	 *
	 * If a Pokémon has not been sent out yet, its remainingHP will be 1.0f.
	 */
	float remainingHP {1.0f};

	/**
	 * Any data attached to the structure
	 */
	std::shared_ptr<UserData> userdata {};

	/**
	 * Pokémon species, as used in Showdown. Different form[e]s
	 * will have different species names, e.g. "Mimikyu" vs "Mimikyu-Busted".
	 *
	 * This string may also end with "*", like "Arceus-*" to indicate that
	 * the Pokémon may match any ID with the same prefix. This usually happens
	 * when the Pokémon must have a form, but that form is currently unknown.
	 */
	std::string species {UNKNOWN};

	/**
	 * Player-selected nickname of the Pokémon
	 */
	std::string nickname {};

	/**
	 * Pokémon's ability.
	 *
	 * If this string has the value "0", "1" or "H", then it's the
	 * ability found in the Pokédex entry for this Pokémon in the
	 * corresponding slot (first, second or hidden ability). Otherwise
	 * this field will contain the ID of the ability as used in Showdown.
	 *
	 * It is to be ignored if the format doesn't support abilities (Generation < III
	 * or Let's Go Pikachu/Eevee)
	 */
	std::string ability {UNKNOWN};

	/**
	 * ID of the last move used by the Pokémon, empty if unknown
	 */
	std::string lastUsedMove {};

	/**
	 * ID of the last move that the Pokémon activated. For example, if it uses
	 * *Metronome* and invokes *Ancient Power*, this will be "ancientpower".
	 * In most cases, though, it will equal lastUsedMove
	 */
	std::string lastActivatedMove {};

	/**
	 * ID of the item the Pokémon is holding as used in Showdown.
	 * Empty string if none, @ref UNKNOWN if unknown.
	 */
	std::string item {UNKNOWN};

	/**
	 * Type of the Pokéball the Pokémon is in. Flair-only, should have no
	 * effect on the actual gameplay. If empty, it's a regular Pokéball.
	 */
	std::string ball {};

	/**
	 * The Pokémon's Individual Values, from 0 to 31 per stat. They will
	 * be known to you if it's your Pokémon, and will be unknown otherwise
	 * (in which case the structure will be filled with -1).
	 *
	 * For Generations I and II, this field will contain Determinant Values
	 * *multiplied by 2*. DVs normally range from 0 to 15, but Showdown always
	 * multiplies them by 2 in the protocol, so they that an IV value divided
	 * by 2 and rounded toward 0 = DV. The library will not work around this quirk
	 * so users can do calculations with IV-based code in some cases.
	 *
	 * Also, in Generations I and II, Special Attack will always equal Special Defense.
	 *
	 * In non-standard formats (e.g. *Hackmons*) there may be different IV
	 * limits.
	 */
	Stats IV;

	/**
	 * The Pokémon's Effort Values, aka Base Points, aka Stat Experience.
	 * If it's not your Pokémon, they will be unknown and the field will
	 * be filled with -1.
	 *
	 * In the core games since Generation III the EVs range from 0 to 255,
	 * and their sum total may not exceed 510.
	 *
	 * In Generations I and II EVs range from 0 to 65536. If you're using a
	 * C++ compiler where ``int`` happens to be smaller, saturation arithmetic
	 * will be used.
	 *
	 * In *Let's Go, Pikachu/Eevee* this field will contain Awakening Values,
	 * which range from 0 to 200.
	 *
	 * In non-standard formats (e.g. *Hackmons*) there may be different EV
	 * limits.
	 */
	Stats EV {};

	/**
	 * Stat boosts (positive or negative) the Pokémon currently has.
	 * They are reset when the Pokémon is switched out. The stat
	 * boosts are represented in stages from -6 to 6 with unchanged
	 * stats being at stage 0.
	 */
	Stats statBoosts {};

	/**
	 * The four moves the Pokémon knows. Unknown moves will have
	 * names equal to @ref MoveSlot::UNKNOWN. MoveSlots with no
	 * names mean that the Pokémon knows less than 4 moves (these
	 * slots, if present, will always be at the end of the array).
	 *
	 * The move *Hidden Power* will have one of the following IDs:
	 * - "hiddenpower" if neither the type nor the power are known
	 * - "hiddenpower<type>" (e.g. "hiddenpowerbug") if only the type is known
	 * - "hiddenpower<type><power>" (e.g. "hiddenpowerbug60") if the type and the power are known
	 */
	std::array<MoveSlot, 4> moves {};

	/**
	 * A list of *volatile* conditions affecting the Pokémon, such as
	 * "confusion" or "taunt". They end when the Pokémon is switched out.
	 *
	 * The name of the condition (always lowercase) is the key. The value
	 * will be 1 in most cases, but conditions like "perish" (the counter
	 * resulting from the move *Perish Song* or the ability *Perish Body*)
	 * or "stockpile" (the counter associated with the eponymous move) may
	 * have other values.
	 *
	 * Note that Perish Song and Perish Body, specifically, will both add
	 * "perishsong": 1 and "perish": <actual counter>.
	 *
	 * "dynamax" is also implemented in Showdown as a volatile condition.
	 */
	std::map<std::string, int> volatiles {};
};

/**
 * Create a JSON representation of a Pokémon team
 *
 * The following fields will be represented in JSON, and these fields have
 * to be filled out with valid non-unknown values:
 *
 * - ability [if generation >= 2]
 * - ball (may be empty) [if generation >= 3]
 * - EV
 * - gender [if generation >= 2]
 * - happiness
 * - item (may be empty) [if generation >= 2]
 * - IV
 * - level
 * - moves (only IDs will be considered)
 * - nature [if generation >= 3]
 * - nickname (may be empty)
 * - species
 * - shiny [if generation >= 2]
 */
nlohmann::json teamToJSON(int generation, const std::vector<Monster>& team);

/// Create a packed representation of a Pokémon team
/// userdata won't be represented
std::string packTeam(const std::vector<Monster>& team);

/// Parse a JSON representation of a Pokémon team and write it into the result vector
void teamFromJSON(const nlohmann::json& json, std::vector<Monster>& result);

/// Parse a packed representation of a Pokémon team and write it into the result vector
void unpackTeam(const std::string& team, std::vector<Monster>& result);


/// An order to a Pokémon in battle
struct Order {
	/**
	 * * 0: no order
	 * * 1 through 4: use move in slot (`action - 1`)
	 * * >= 5: switch to Pokémon with the 1-based team index (`action - 4`)
	 */
	unsigned int action;

	/// Use this modifier (to be ignored if the action is not a move)
	MoveModifier modifier;
};


// ===== BATTLE EVENTS =====
struct BattleEvent {
	/// The animation is to be suppressed
	bool noAnimation;

	/// Show no message
	bool noMessage;

	/// Event type
	const int type;

	/**
	 * If `fromEffect` is empty, `fromPosition` will be 0. Otherwise it will be
	 * the position that caused the effect, 0 if unknown.
	 */
	int effectFromPosition;

	/**
	 * fromEffect, fromEffectCategory and alsoFromMove may contain the reason
	 * why the event happened, as reported by Showdown. All three will often
	 * be empty.
	 *
	 * If the report from Showdown contains a colon, like ``[from] ability: Cursed Body``,
	 * what is before the first colon (``ability``) will be put into `fromEffectCategory`,
	 * and what's after it (``Cursed Body``) will be put into `fromEffect`. Otherwise
	 * `fromEffectCategory` will be empty and everything will be put into ``fromEffect``.
	 * The strings won't start or end with spaces.
	 *
	 * `alsoFromMove` will be filled in if Showdown sends something like `[from] stealeat [move] Pluck`,
	 * and the value will be that of the ``[move]`` tag (i.e. ``Pluck`` in this example).
	 * This seems to only happen when ``fromEffect == "eat"`` or ``fromEffect == "stealeat"``,
	 * and this, in turn, happens when a berry gets eaten due to a move.
	 */
	std::string fromEffectCategory;
	std::string fromEffect;
	std::string alsoFromMove;
};

/**
 * A Pokémon used a move.
 *
 * Unless the move misses (in which case `miss` will be true), the outcome
 * of it can only be deduced from future events.
 */
struct MoveUseEvent : public BattleEvent {
	/// Whether the move missed
	bool miss;

	/**
	 * Whether the move was used as a Z-Status Move. If so,
	 * it may have had some additional effects.
	 *
	 * Z-Attacks have their own names (`withZEffect` will be false).
	 */
	bool withZEffect;

	/// What position the move came from.
	int userPosition;

	/**
	 * What position the move targeted.
	 *
	 * If the move has multiple targets or no target, ``targetedPosition``
	 * may be 0 or any other value and should be ignored. If the move targets
	 * a side, the target can be any position, possibly even a fainted Pokémon,
	 * on that side.
	 */
	int targetedPosition;

	/// Name of the move
	std::string name;

	/**
	 * Animation to be displayed. If empty, display the move's own animation,
	 * unless ``noAnimation`` is true.
	 */
	std::string animateAs;
};


/// A Pokémon was switched out for another one
struct SwitchEvent : public BattleEvent {
	/**
	 *  If true, the new Pokémon was dragged out rather than selected
	 *  by its player (e.g. by the move *Whirlwind*)
	 */
	bool drag;
	/// The position in question
	int position;
	/// Exact HP of the new Pokémon, -1 if unknown
	int HP;
	/// Status condition of the new Pokémon
	NVStatus status;
	/// Team index of the Pokémon that switched out
	unsigned int oldTeamIndex;
	/// Team index of the new Pokémon
	unsigned int newTeamIndex;
	/// Remaining HP fraction of the new Pokémon, ranging from 0.0f to 1.0f
	float remainingHP;
};


/// A Pokémon lost or gained HP and possibly got a new non-volatile status
/// The stat boosts on one or two Pokémon have changed
struct BoostChangeEvent : public BattleEvent {
	enum class Type {
		/// Boosts added/removed on one Pokémon (newBoosts deduced by library)
		ALTER,
		/// Boosts set to specific values on one Pokémon (changes deduced)
		SET,
		/// Reset all boosts to 0 on one Pokémon
		/// changes and affectedStats deduced
		CLEAR,
		/// Reset all positive boosts to 0 on one Pokémon
		/// changes and affectedStats deduced
		CLEAR_POSITIVE,
		/// Reset all negative boosts to 0 on one Pokémon
		/// changes and affectedStats deduced
		CLEAR_NEGATIVE,
		/// Invert all boosts (i.e. multiply by -1) on one Pokémon.
		/// changes and new values deduced.
		INVERT,
		/// Copy some boosts form source to target
		/// changes and newBoosts will be deduced
		COPY,
		/// Swap some boosts between target and source.
		/// changes, newBoosts, sourceChanges and sourceNewBoosts will be deduced
		SWAP,
	};

	/// The target Pokémon or the only affected Pokémon
	int position;

	/// The source Pokémon, 0 if only one was affected
	int sourcePosition;

	/// In this field stats that have changed will be set to 1, others to 0
	Stats affectedStats;

	/// Changes to the target Pokémon (difference between previous boost and the new boost)
	Stats changes;

	/// The resulting new stat boosts of the target Pokémon
	Stats newBoosts;

	/// Changes to the source Pokémon, nonsense values if sourcePosition == 0
	Stats sourceChanges;

	/// New boosts of the source Pokémon, nonsense values if sourcePosition == 0
	Stats sourceNewBoosts;
};


/// Represents a number of events of simple structure with a text field
struct DamageHealStatusEvent : public BattleEvent {
	/**
	 * If this is true, the server has forced a particular HP value
	 * for the Pokémon, the rest was deduced from the previous situation.
	 */
	bool setHP;

	/// Whether the status has changed
	bool hasStatusChanged;

	/// Whether the HP has changed
	bool hasHPChanged;

	/**
	 * The new non-volatile status. Will equal the status the Pokémon
	 * had before if it hasn't changed.
	 */
	NVStatus status;

	/// The Pokémon's position
	int position;

	/**
	 * The change in HP. If negative, the Pokémon suffered damage; if positive,
	 * the Pokémon was healed. It may be 0, and it will be 0 if the exact change
	 * in HP is unknown.
	 */
	int HPChange;

	/// Exact new HP. Will be -1 if unknown.
	int newHP;

	/// Exact new maximum HP (usually unchanged). Will be -1 if unknown.
	int newMaxHP;

	/**
	 * HP change as a fraction from 0 to 1. This may be non-zero even if
	 * `HPChange` is 0.
	 */
	float remainingHPChange;

	/// Remaining HP fraction from 0 to 1.
	float remainingHP;
};


struct MiscellaneousTextEvent : public BattleEvent {
	enum class Type {
		/**
		 * A Pokémon's item has been changed or revealed
		 *
		 * `position` will be the Pokémon's position
		 * `text` will be the name of the (new) item
		 *
		 * WARNING: see the comment on ITEM_DESTROYED
		 */
		ITEM,

		/**
		 * The Pokémon's item has been destroyed, and it's
		 * currently holding no item
		 *
		 * `position` will be the Pokémon's position
		 * `text` will be the name of the item that was destroyed
		 *
		 * WARNING: this event will not be emitted when the item is transferred
		 * to another Pokémon (e.g. by moves like *Thief* or *Bestow*).
		 * Consequently, the library will not update the Pokémon's data.
		 *
		 * The ITEM event will be emitted for the receiving Pokémon, though,
		 * so when that event is emitted, you should look up `fromEffect`
		 * in a move database and if the effect causes items to be transferred,
		 * set the Pokémon's item appropriately yourself. However, if you don't,
		 * the only consequence will be obsolete item information in future
		 * BattleStates, no errors should occur.
		 */
		ITEM_DESTROYED,

		/**
		 * Same as ITEM_DESTROYED—with all the caveats—when the item is
		 * a berry and it was eaten by the Pokémon.
		 *
		 * Note that sometimes ITEM_DESTROYED may be emitted instead,
		 * even if it was a berry that was eaten, depending on what
		 * Showdown sends.
		 */
		ITEM_EATEN,

		/**
		 * A volatile effect has been added to a Pokémon
		 *
		 * `position` will be the Pokémon's position
		 * `text` will be the ID of the volatile effect, as used in Showdown
		 */
		VOLATILE_START,

		/**
		 * A volatile effect on a Pokémon has ceased. Otherwise like
		 * VOLATILE_START
		 */
		VOLATILE_END,

		/**
		 * An ability has been activated that can't be better described
		 * by a different event.
		 *
		 * `position` will be the position where the Pokémon with the ability is
		 * `text` will be the name of the ability
		 */
		ABILITY,
	};
};
// ===== END OF BATTLE EVENTS =====


/// An exception class for when an order is invalid
class InvalidOrderError : public std::exception {
public:
	InvalidOrderError(const Order& order, int position, const std::string& reason);

	/// Get the order in question (allocated within the structure)
	const Order& getOrder() const;
	/// Get the position for which the order was given
	int getPosition() const;
	/// Get the reason for failure
	const std::string& getReason() const;
protected:
	int position;
	Order order;
	std::string reason;
};


/// An exception class for when an order is, for example, too late
class InvalidBattleStateError : public std::exception {
};


/// Information about a Pokémon visible on the screen, used to
/// identify the Pokémon
struct MonsterDetails {
	/// Parse from Showdown representation
	static MonsterDetails fromString(const std::string& details);

	std::string species;
	bool shiny;
	Gender gender;
	int level;

	MonsterDetails(const std::string& species = "", bool shiny = false, Gender gender = Gender::NONE, int level = 100) :
		species(species), shiny(shiny), gender(gender), level(level) {
	}
};


/// HP of a Pokémon
struct MonsterHP {
	/// Parse from Showdown representation. If should_be_exact is true, assume the HP is
	/// exact rather than use heuristics
	static MonsterHP fromString(const std::string& details, bool should_be_exact = false);

	/// Exact current HP; less than 0 if unknown
	int current;
	/// Exact maximum HP; less than 0 if unknown
	int max;
	/// Remaining HP as a ratio
	float remainingHP;
};


/**
 * Representation of the state of a battle and a recording of orders
 *
 * Battle interaction through libPShowdown mostly occurs in form of
 * receiving BattleState instances from library calls, adding orders
 * and sending the instance back. You may also construct instances
 * and use them yourself without involving other library operations,
 * if you need a representation of battle state for your own purposes.
 */
class BattleState {
public:
	/// General "category" of the battle
	enum class Category {
		/**
		 * Battle between two players with 1 Pokémon on each side.
		 *
		 * The only valid positions are 1 (your Pokémon) and -1
		 * (the opposing Pokémon). Positions are ignored most of the time,
		 * and all moves select targets automatically.
		 */
		SINGLES,

		/**
		 * Battle between two players with 2 Pokémon on each side.
		 *
		 * Valid positions are 1 (your first Pokémon), 2 (your
		 * second Pokémon), -1 (opposing Pokémon facing 1) and -2 (opposing
		 * Pokémon facing 2).
		 */
		DOUBLES,

		/**
		 * Battle between two players with 3 Pokémon on each side.
		 *
		 * Your Pokémon will be in the positions 1, 2 and 3, while
		 * the opposing Pokémon facing them will be in -1, -2 and -3, respectively.
		 *
		 * Adjacency rules will be in play: 1, -1, 2 and -2 are adjacent to
		 * each other, and 3, -3, 2 and -2 are adjacent to each other (2 and -2
		 * are thus adjacent to all others).
		 *
		 * If only one Pokémon is left on each side (with no more in either team),
		 * they will eventually be moved to positions 1 and -1, respectively.
		 */
		TRIPLES,

		/**
		 * 2 players vs 2 players (2 Pokémon on each side but
		 * each is controlled by a different player).
		 *
		 * Valid positions are 1 (your Pokémon), 2 (your ally's Pokémon),
		 * -1 and -2 (opposing Pokémon facing you, respectively).
		 */
		MULTI,

		/**
		 * 4-player free for all (4 sides with 1 Pokémon on each
		 * of them).
		 *
		 * Your Pokémon will be in position 1 while your opponents
		 * will be controlling positions -1, -2 and -3. For visualization purposes,
		 * -2 is on your left, -3 is on your right, -1 faces you. In terms of
		 * game mechanics, all Pokémon are adjacent to each other.
		 */
		FREE_FOR_ALL,
	};

	/**
	 * Return the corresponding category for the given case-insensitive name ("singles",
	 * "doubles", "triples", "multi" or "free-for-all"; ' ', '-' and '_' ignored).
	 *
	 * @throw std::out_of_range if no such category
	 */
	static Category categoryByName(const std::string& name);

	/// Outcome of a battle
	enum class Outcome {
		/// The battle is still on
		ONGOING = 0,

		/// You won
		VICTORY = 0x10,

		/// You won, opponent forfeited or left
		VICTORY_OPPONENT_FORFEITED = 0x12,

		/// You won by timeout
		VICTORY_BY_TIMEOUT = 0x13,

		/// You won by tie resolution
		VICTORY_BY_RESOLUTION = 0x14,

		/// You lost
		DEFEAT = 0x20,

		/// You forfeited or left
		DEFEAT_FORFEITED = 0x21,

		/// You lost by timeout
		DEFEAT_BY_TIMEOUT = 0x22,

		/// You lost by resolution
		DEFEAT_BY_RESOLUTION = 0x23,

		/// Tie
		TIE = 0x30,
	};

	/// What kind of orders (if any) the battle needs from the player
	enum class Request {
		/// Nothing
		NONE,

		/// The battle is in team preview and awaits .selectTeam
		SELECT_TEAM,

		/// Select one or more Pokémon to send out with .orderSwitch
		SELECT_MONSTER,

		/// Give turn orders with .orderSwitch or .orderUseMove
		TURN,
	};

	using timestamp_t = std::chrono::time_point<std::chrono::steady_clock>;

	/**
	 * Construct a BattleState with the given category, generation, ID
	 * and timestamp.
	 *
	 * @param    category      The appropriate Category value. Cannot be
	 *     changed later.
	 * @param    generation    GENERATION_MIN through GENERATION_MAX. Cannot be changed later.
	 * @param    ID            Any value. It is not used internally, but
	 *     can be retrieved and set for callers' purposes, e.g. to distinguish
	 *     different ongoing battles.
	 * @param    timestamp     Treated the same way as ID.
	 */
	BattleState(
			Category category = Category::SINGLES,
			unsigned int generation = GENERATION_MIN,
			size_t ID = 0,
			timestamp_t timestamp = {});

	BattleState(const BattleState& other);
	BattleState(BattleState&& other);
	BattleState& operator=(const BattleState& other);
	BattleState& operator=(BattleState&& other);

	Category getCategory() const noexcept;
	unsigned int getGeneration() const noexcept;
	size_t getID() const noexcept;
	timestamp_t getTimestamp() const;
	/// Discover what the Showdown server expects you to do
	Request getRequest() const;
	/// Get the set of standard rules of the battle (as flags)
	unsigned long getRules() const;
	/// Get the set of rules that couldn't be parsed normally. It's allocated within the object.
	const std::set<std::string>& getNonstandardRules() const;
	/// Return previously attached user data, if any
	std::shared_ptr<UserData> getUserData() const;
	Outcome getOutcome() const;

	/// Set user data for the battle state
	void setUserData(std::shared_ptr<UserData> user_data);

	/// Get the maximum number of Pokémon per team during team preview
	/// (or during the game, if no team preview in the format).
	/// Return 0 if not yet set.
	unsigned int getInitialTeamSize() const;

	/// Get the number of Pokémon that can battle. Greater or equal to
	/// the initial team size.
	unsigned int getBattlingTeamSize() const;

	/**
	 * Get a pointer to a given Pokémon on the team of the player
	 * controlling the given position. If the Pokémon
	 * should exist but nothing is known about it, it will still be
	 * a valid pointer to an appropriately-filled structure. If
	 * the Pokémon doesn't exist at all, it will be a nullptr.
	 *
	 * In SINGLES, DOUBLES and TRIPLES, query your team if `position`
	 * is any value > 0, the opposing team if `position` < 0, return `nullptr`
	 * if `position == 0`. In other formats `position` must be a valid
	 * position number, or else the method will always return `nullptr`.
	 *
	 * Remember that you always control at least position 1.
	 *
	 * @param    position      position number
	 * @param    team_index    1-based index of the Pokémon in the team.
	 *     May be 0 to mean the active Pokémon in that position, but then
	 *     the position must be valid (even in SINGLES, DOUBLES and TRIPLES),
	 *     and there must be a Pokémon there. If the team_index makes no sense,
	 *     the method will return `nullptr`.
	 *
	 * @return Pointer to the Monster structure within the BattleState or
	 * `nullptr`. The structure will exist as long as the BattleState does.
	 */
	Monster* getMonster(int position, unsigned int team_index) noexcept;
	const Monster* getMonster(int position, unsigned int team_index) const noexcept;

	/**
	 * Get the 1-based team index of the Pokémon that's currently in the
	 * controlled position, suitable for use with @ref getMonster.
	 * Return 0 if the position doesn't exist or there is no Pokémon in it.
	 */
	unsigned int getTeamIndexAt(int position) const noexcept;

	/**
	 * Select Pokémon from your team that will actually battle. This is
	 * only possible when @ref getRequest returns SELECT_TEAM.
	 *
	 * The `Iterator` class must be such that ``++start`` is well-formed,
	 * ``static_cast<int>(*start)`` is well-formed and returns a number
	 * from 1 to getInitialTeamSize(). The entire sequence must be 1-based indices
	 * of Pokémon that will participate in the battle. The size of the sequence
	 * must equal the value returned by @ref getBattlingTeamSize.
	 *
	 * After you send this order, the following BattleStates you receive will have
	 * team indices rearranged according to the order you set, non-participating
	 * Pokémon will be removed from the team.
	 *
	 * The information from Team Preview will be available to you before you
	 * select a team. Other players' teams will be rearranged by moving their
	 * leading Pokémon to the front; there will be no changes otherwise.
	 * Once an opponent sends the maximum number of Pokémon they may have,
	 * the info on their non-participating Pokémon may be dropped by the library.
	 *
	 * The first one, two or three Pokémon (in SINGLES, DOUBLES and TRIPLES,
	 * respectively) in the order of selection will be sent out first.
	 *
	 * If the format requires participation with a full team (i.e.
	 * ``getInitialTeamSize() == getBattlingTeamSize()``), you may instead
	 * call @ref orderSwitch for every position. Selected team indices will be
	 * moved to the front. Otherwise you must call this method.
	 *
	 * @throw    InvalidOrderError    with position == 0 if:
	 *     * the request is not SELECT_TEAM, or
	 *     * ``*start`` returns an invalid or repeated team index at any point
	 */
	template<class Iterator> void selectTeam(Iterator start);

	/**
	 * Order the Pokémon in `position` to be replaced with the one
	 * with `new_team_index`.
	 *
	 * Will always throw an InvalidOrderError if:
	 *     * `position` is not controlled by you, or
	 *     * `new_team_index` is invalid or no Pokémon is there, or
	 *     * the request is NONE, or
	 *     * the request is SELECT_TEAM and ``getInitialTeamSize() != getBattlingTeamSize()``
	 *
	 * Unless `force` is true, will also throw an InvalidOrderError if:
	 *     * the Pokémon at `new_team_index` has fainted, or
	 *     * it is already on the field
	 *
	 * When using this method when the request is SELECT_TEAM, switch order won't be
	 * recorded. The team will be rearranged instead.
	 *
	 * @return Team index of the Pokémon currently in that position or 0 if none
	 */
	unsigned int orderSwitch(int position, unsigned int new_team_index, bool force = false);

	/**
	 * Order the Pokémon in `position` to use the move in `move_slot`
	 * (1-based index, to be consistent with how the class works), and
	 * with the appropriate modifier.
	 *
	 * Will always throw an InvalidOrderError if:
	 *     * `position` is not controlled by you, or
	 *     * `slot` is not between 1 and 4, or
	 *     * the request is not TURN, or
	 *     * `getGeneration() < 6` and `modifier` is not `NONE`, or
	 *     * `getGeneration() < 7` and `modifier` is `Z` or `DYNAMAX`, or
	 *     * `getGeneration() < 8` and `modifier` is `DYNAMAX`
	 *
	 * Unless `force` is true, will also throw an InvalidOrderError if:
	 *     * there is no Pokémon in that position, or
	 *     * there is no move in `slot`, or
	 *     * the `move` in `slot` is disabled
	 *
	 * While this method will not prevent you from trying to Mega-Evolve or
	 * use a Z-Move in Generation VIII, the games and standard formats on Showdown
	 * do not allow it.
	 */
	void orderUseMove(int position, unsigned int slot, MoveModifier modifier, bool force = false);

	/**
	 * Order the Pokémon in position 1 or 3 to switch places with position 2.
	 * This is only possible in TRIPLES.
	 *
	 * @throw InvalidOrderError if:
	 * * the position is invalid or it isn't TRIPLES, or
	 * * there is no Pokémon in the selected position
	 */
	void order3BShift(int position);

	/**
	 * Update the BattleState with the data from a Showdown request (as serialized JSON)
	 */
	void _registerRequest(const std::string& request);

	/// Register a switch event
	void _registerSwitch(bool drag, int position, const MonsterDetails& new_monster, const MonsterHP& new_monster_hp);
};


/**
 * An abstract base for classes capable of processing battle events.
 *
 * When writing a bot, you'll probably want to make the interface to that
 * bot a subclass of BattleListener, which stores a pointer to the corresponding
 * BattleServerConnection (which will be a @ref DirectSimConnection or a @ref ShowdownServerConnection).
 * Make an object of your bot's class and either plug it directly into the BattleServerConnection
 * (usually not recommended, as it might lead to a deadlock or the network connection
 * breaking up) or through a @ref BattleListenerDispatcher. Your bot must then send
 * orders directly to the BattleServerConnection.
 */
class BattleListener {
public:
	virtual ~BattleListener() = default;

	/**
	 * The server requests that you send orders for the battle. The exact
	 * request must be read from the battle state.
	 */
	virtual void requestOrders(std::unique_ptr<BattleState> battle_state) = 0;

	/**
	 * The server rejected your orders as invalid. You are expected to send
	 * corrected orders.
	 *
	 * @param    battle_state    This may be the unchanged battle state or a new battle
	 *     state, if the server sent one. Your orders, however, will be copied over,
	 *     unchanged.
	 * @param    errors          Error messages for positions 1, 2 and 3, respectively.
	 *     An empty string means there was no error for that position.
	 */
	virtual void requestCorrectedOrders(std::unique_ptr<BattleState> battle_state,
			std::array<std::string, 3> errors) = 0;

	/**
	 * The battle has ended. You are receiving a final BattleState update.
	 *
	 * You are to call sendOrders on your BattleServerConnection (without actually
	 * setting any orders) so it could repurpose the BattleState object and free its
	 * own buffers. Not doing so may result in a memory leak.
	 */
	virtual void endBattle(std::unique_ptr<BattleState> final_battle_state) = 0;
protected:
	BattleListener() = default;
};


/**
 * The BattleListenerDispatcher maintains a pool of worker threads and has
 * these threads call attached BattleListeners.
 *
 * .requestOrders, .requestCorrectedOrders, .endBattle and .message of a BattleListenerDispatcher
 * will queue a corresponding event and return as soon as possible. Then, once a worker
 * thread becomes available, it will call the appropriate method of the
 * associated BattleListener. If .requestCorrectedOrders needs to be called, it
 * gets a priority over the other three methods.
 *
 * The dispatcher decides what listener to send updates to based on the ID of the
 * BattleState.
 *
 * Instances of this class must be created with the static method @ref create.
 */
class BattleListenerDispatcher : public BattleListener {
public:
	static std::shared_ptr<BattleListenerDispatcher> create(size_t workers, size_t max_attached_listeners);

	BattleListenerDispatcher(BattleListenerDispatcher&) = delete;
	BattleListenerDispatcher(BattleListenerDispatcher&&) = delete;
	virtual ~BattleListenerDispatcher();

	/**
	 * Attach a new listener and register that BattleStates with this ID must
	 * be sent to it. The ID must not be 0.
	 *
	 * @throw std::invalid_argument if there is another already another listener
	 *     associated with this battleID or if the ID is 0
	 * @throw std::runtime_error if this dispatcher already has the maximum number
	 *     of listeners.
	 */
	void attachListener(size_t battleID, std::shared_ptr<BattleListener> listener);

	/**
	 * Like attachListener but for multiple listeners at once.
	 *
	 * The BattleIDIterator must dereference as size_t, SPtrBattleListenerIterator
	 * must dereference as std::shared_ptr<BattleListener>.
	 *
	 * If an exception noted in attachListener occurs, none of the listeners will have
	 * been attached.
	 */
	template<class BattleIDIterator, class SPtrBattleListenerIterator>
	void attachListeners(BattleIDIterator BIDs_start, BattleIDIterator BIDs_end,
			SPtrBattleListenerIterator listeners_start);

	/**
	 * Reserve space for this many BattleListeners. You don't have to call this
	 * method for attachListener or attachListeners to succeed.
	 *
	 * If this method returns true, the space will have successfully been reserved.
	 * Attaching `number` more listeners will not cause a runtime error due to
	 * the maximum number being reached.
	 *
	 * If this method returns false, there are too many listeners already attached
	 * and the space will not be reserved.
	 *
	 * When a listener is attached, the reserved space, if any, is used first.
	 */
	bool reserveListenerSlots(size_t number);

	/**
	 * Remove reservations for `number` listeners. If the number is bigger than
	 * the number of reserved listener slots, or if the number equals 0, all
	 * reserved listener slots will be freed.
	 *
	 * @return The number of slots that were freed, less or equal to `number`
	 *     (unless `number` is 0)
	 */
	size_t freeListenerSlots(size_t number);

	/**
	 * Remove the listener associated with the battle ID. If `reserve` is true,
	 * also reserve one listener slot (the one formerly used by the listener).
	 *
	 * @throw std::out_of_range if no listener associated with the ID
	 */
	void removeListener(size_t battleID, bool reserve = false);

	/**
	 * Remove multiple listeners. If `reserve` is true, reserve every slot that's
	 * being freed.
	 *
	 * BattleIDIterator must dereference as size_t.
	 *
	 * @throw std::out_of_range if no listener associated with the ID. If this exception
	 * is thrown, no listeners will have been removed
	 */
	template<class BattleIDIterator>
	void removeListeners(BattleIDIterator BIDs_start, BattleIDIterator BIDs_end,
			bool reserve = false);

	virtual void requestOrders(std::unique_ptr<BattleState> battle_state) override;
	virtual void requestCorrectedOrders(std::unique_ptr<BattleState> battle_state,
				std::array<std::string, 3> errors) override;
	virtual void endBattle(std::unique_ptr<BattleState> final_battle_state) override;
};


/**
 * An abstract base for classes that relay orders to Pokémon Showdown servers or
 * simulator processes. It is subclassed within the library and not necessarily
 * designed to be subclassed by users.
 */
class BattleServerConnection {
public:
	virtual ~BattleServerConnection();

	/**
	 * Send orders attached to the provided BattleState to the server.
	 *
	 * @throw InvalidBattleStateError if `bs_with_orders` either corresponds
	 *     to a non-existent battle or the battle has moved on (e.g. due to a timeout)
	 * @throw InvalidOrderError if the orders in `bs_with_orders` can't be processed
	 * @throw std::runtime_error if the orders can't be submitted (e.g. the connection
	 *     is closed)
	 *
	 * Note that it is possible that no exception will be thrown, but the server
	 * will reject the orders as invalid. In that case the BattleServerConnection
	 * will call requestCorrectedOrders on the corresponding BattleListener.
	 */
	virtual void sendOrders(std::unique_ptr<BattleState> bs_with_orders) = 0;
};


/**
 * This class is used to interact with a simulator running on the local computer.
 * An instance can be created with the static create method.
 *
 * On POSIX systems, run the simulator by launching ``pokemon-showdown simulate-battle``,
 * then pass stdin and stdout file descriptors of the child process to the POSIX_create.
 * This method doesn't exist if LIBPSHOWDOWN_POSIX is not defined.
 *
 * If LIBPSHOWDOWN_NO_DIRECTSIM is defined, then this class will still exist, but the *create()
 * methods will not, so it won't be usable.
 *
 * After that, call .runBattle
 */
class DirectSimConnection : public BattleServerConnection {
public:
	using t_listeners = std::array<std::pair<std::shared_ptr<BattleListener>, size_t>, 4>;

#ifdef LIBPSHODOWN_POSIX
	static std::shared_ptr<DirectSimConnection> POSIX_create(int sim_input_fd, int sim_output_fd);
#endif

	DirectSimConnection(DirectSimConnection&) = delete;
	DirectSimConnection(DirectSimConnection&&) = delete;

	/**
	 * Run a battle in the given format with the given teams. If there is
	 * already an ongoing battle, it will be reset (endBattle on listeners
	 * won't be called). Pipes to the simulator must already have been attached.
	 *
	 * This method will not return unless and until:
	 * - an exception occurs (in which case the battle can't continue), or
	 * - another thread calls runBattle and so starts a new battle, or
	 * - stopBattle is called (from the same thread or another one), or
	 * - the battle ends, endBattle is called on every listener, and each of them
	 *     calls sendOrders to confirm they have received the information
	 *
	 * When this method returns, including via an exception, this object will no longer
	 * have any pointers to the listeners.
	 *
	 * The listeners will be called from the same thread. If they are BattleListenerDispatchers,
	 * the dispatcher(s) will, of course, dispatch the calls through their worker threads.
	 *
	 * @param    format           Format of the battle to send to Showdown (e.g. "gen7ou"
	 *     or "gen7randombattle"). Category will be received from Showdown.
	 * @param    generation       Generation of the format. If a different generation is reported
	 *     by Showdown, an exception will be thrown.
	 * @param    initial_teams    Players' initial teams. Vectors for the third and the fourth
	 *     players will be ignored if it isn't a format with 4 players. The structures must be
	 *     suitable for @ref packTeam. If a team is expected but isn't provided (i.e. the vector
	 *     is empty), an empty team will be sent to Showdown; this is what you should do when starting
	 *     a random battle (if you send your own team for a random battle, it will be ignored).
	 * @param    listeners        BattleListeners to attach and IDs to assign to them.
	 *     IDs must not be 0 and must not repeat, otherwise they can be chosen freely.
	 *     Pairs for the 3rd and the 4th players will be ignored if it isn't a format
	 *     with four players.
	 *
	 * If the arguments are invalid in a way the library can't detect, the simulator will
	 * be in an undetermined state. It doesn't send error messages in response to client
	 * errors, so the library will be expecting it to send messages, while it will be
	 * waiting for the library to send correct commands. This deadlock can only be resolved
	 * by calling stopBattle. The simulator might also just abort.
	 *
	 * If this method throws an exception, it is best to destroy the object, close the pipes
	 * and run a new instance of the simulator.
	 *
	 * The function is neither reentrant nor can run in parallel with itself.
	 */
	void runBattle(const std::string& format,
			int generation,
			std::array<std::vector<Monster>, 4>& initial_teams,
			t_listeners& listeners);

	/**
	 * Stop the ongoing battle, if any, and cause runBattle to returns. Can be
	 * called from the thread in runBattle or from another thread. This function
	 * only queues the stopping of a battle and forces runBattle to stop waiting
	 * for data from the simulator (if it is), and then returns.
	 *
	 * @return true if a battle has been stopped, false if no battle was running
	 *     or if termination has already been queued.
	 */
	bool stopBattle();

	/**
	 * Have all messages sent and received logged into a stream. Every message will be prefixed by
	 * the `prefix`, folloed immediately by:
	 * - "SEND: " if sent
	 * - "RECV: " if received
	 * - "RVPL: " (received partial line): if received and the original data didn't end with '\n'
	 * - "EROR: " if an error message logged by the library itself
	 * - "INFO: " if an informational message logged by the library itself
	 *
	 * If `stream` is `nullptr`, logging will be disabled.
	 *
	 * The stream must continue to exist until a new log stream is attached, the log stream
	 * is removed or this object is destroyed.
	 *
	 * This function can be called in parallel with other methods.
	 */
	void setLogStream(std::ostream* stream, const std::string& prefix);

	virtual void sendOrders(std::unique_ptr<BattleState> bs_with_orders) override;
};


/**
 * This is also a direct connection to the simulator, but is used for things like
 * generating teams rather than battling. It requires a special launcher from
 * the Seimei AI Project.
 *
 * Alternatively, you can generate teams by running ``pokemon-showdown generate-team <format>``,
 * and reading its output—one line, which should be the team represented in the *packed format*.
 * This will require a few seconds of start-up time, though.
 *
 * Unlike DirectSimConnection, it is synchronous: every method writes data to the
 * simulator's input, reads the output and returns the result. It does, however, internally
 * use a mutex so that multiple threads could call its methods at the same time.
 */
class SeimeiAuxSimConnection {
public:
#ifdef LIBPSHOWDOWN_POSIX
	static std::shared_ptr<SeimeiAuxSimConnection> POSIX_create(int sim_input_fd, int sim_output_fd);
#endif

	SeimeiAuxSimConnection(SeimeiAuxSimConnection&) = delete;
	SeimeiAuxSimConnection(SeimeiAuxSimConnection&&) = delete;

	std::vector<Monster> generateTeam(const std::string& format);
};
}
//@formatter:on

#include <libPShowdown/internal.tpp>

#endif
