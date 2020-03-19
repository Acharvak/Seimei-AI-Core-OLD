#define LIBPSHOWDOWN_POSIX
#include <libPShowdown.hpp>
#include <iostream>
#include <vector>

using namespace pshowdown;
using namespace std;

int main() {
	Monster m1;
	m1.species = "Pheromosa";
	m1.item = "expertbelt";
	m1.ability = "beastboost";
	m1.moves.at(0).ID = "icebeam";
	m1.moves.at(1).ID = "poisonjab";
	m1.moves.at(2).ID = "uturn";
	m1.moves.at(3).ID = "highjumpkick";
	m1.nature = Nature::SERIOUS;
	for(int i {0}; i <= Stats::MAX_INDEX; ++i) {
		m1.EV.at(i) = 85;
		m1.IV.at(i) = 31;
	}
	m1.gender = Gender::NONE;
	m1.level = 78;
	m1.happiness = 255;

	Monster m2;
	m2.species = "Alomomola";
	m2.item = "leftovers";
	m2.ability = "regenerator";
	m2.moves.at(0).ID = "knockoff";
	m2.moves.at(1).ID = "protect";
	m2.moves.at(2).ID = "toxic";
	m2.moves.at(3).ID = "scald";
	m2.nature = Nature::SERIOUS;
	for(int i {0}; i <= Stats::MAX_INDEX; ++i) {
		m2.EV.at(i) = 85;
		m2.IV.at(i) = 31;
	}
	m2.gender = Gender::MALE;
	m2.level = 82;
	m2.happiness = 255;

	std::vector<Monster> team {m1, m2};

	nlohmann::json result {
		{"formatid", "gen7randombattle"},

	};
}

