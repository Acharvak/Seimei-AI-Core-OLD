#!/usr/bin/env python3
# This file is part of the Seimei AI Project:
#     https://github.com/Acharvak/Seimei-AI
#
# Copyright 2020 Fedor Uvarov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This will generate libPShowdown/tries.tpp, which will
# contain character tries used for text parsing.
from __future__ import print_function
import sys

if sys.version_info[0] < 3 \
        or (sys.version_info[0] == 3 and sys.version_info[1] < 3):
    print('This script needs Python 3.3 or later, you have ' \
          + str(sys.version_info[0]) + '.' + str(sys.version_info[1]),
          file=sys.stderr)
    sys.exit(1)


def trie_from_strings(value_prefix, transform, slist):
    if transform is None:
        transform = lambda x: x
    return [(transform(s), value_prefix + s) for s in slist]
    


LIST_OF_TRIES = ['BATTLE_CATEGORIES', 'NATURES', 'NVSTATUS_NAMES', 'PLAYER_IDS',
                 'RULE_NAMES', 'SERVER_REPLY_HEADERS', 'STAT_NAMES', 'TYPE_NAMES',
                 'TRANSLATE_POSITION_DOUBLES_P1', 'TRANSLATE_POSITION_DOUBLES_P2',
                 'TRANSLATE_POSITION_TRIPLES_P1', 'TRANSLATE_POSITION_TRIPLES_P2',
                 'TRANSLATE_POSITION_MULTI_P1', 'TRANSLATE_POSITION_MULTI_P3',
                 'TRANSLATE_POSITION_MULTI_P2', 'TRANSLATE_POSITION_MULTI_P4']

BATTLE_CATEGORIES = trie_from_strings('BattleState::Category::', lambda s: s.lower(), [
    'SINGLES', 'DOUBLES', 'TRIPLES', 'MULTI'
]) + [('freeforall', 'BattleState::Category::FREE_FOR_ALL')]

NATURES = trie_from_strings('Nature::', lambda s: s.lower(), [
    'HARDY', 'LONELY', 'ADAMANT', 'NAUGHTY', 'BRAVE',
    'BOLD', 'DOCILE', 'IMPISH', 'LAX', 'RELAXED',
    'MODEST', 'MILD', 'BASHFUL', 'RASH', 'QUIET',
    'CALM', 'GENTLE', 'CAREFUL', 'QUIRKY', 'SASSY',
    'TIMID', 'HASTY', 'JOLLY', 'NAIVE', 'SERIOUS',
])

NVSTATUS_NAMES = [
    ('fainted', 'NVStatus::FAINTED'), ('fnt', 'NVStatus::FAINTED'),
    ('burn', 'NVStatus::BURN'), ('brn', 'NVStatus::BURN'),
    ('freeze', 'NVStatus::FREEZE'), ('frozen', 'NVStatus::FREEZE'), ('frz', 'NVStatus::FREEZE'),
    ('paralysis', 'NVStatus::PARALYSIS'), ('par', 'NVStatus::PARALYSIS'),
    ('poison', 'NVStatus::POISON'), ('psn', 'NVStatus::POISON'),
    ('toxic', 'NVStatus::TOXIC'), ('tox', 'NVStatus::TOXIC'),
    ('sleep', 'NVStatus::SLEEP'), ('slp', 'NVStatus::SLEEP'),
]

PLAYER_IDS = [('p1', 1), ('p2', 2), ('p3', 3), ('p4', 4)]

RULE_NAMES = trie_from_strings('', lambda s: s[5:].lower().replace('_', ' '), [
    'RULE_2_ABILITY_CLAUSE', 'RULE_3_BATON_PASS_CLAUSE', 'RULE_ACCURACY_MOVES_CLAUSE',
    'RULE_BATON_PASS_CLAUSE', 'RULE_CFZ_CLAUSE', 'RULE_DYNAMAX_CLAUSE',
    'RULE_ENDLESS_BATTLE_CLAUSE', 'RULE_EVASION_ABILITIES_CLAUSE', 'RULE_EVASION_MOVES_CLAUSE',
    'RULE_EXACT_HP_MOD', 'RULE_FREEZE_CLAUSE_MOD', 'RULE_HP_PERCENTAGE_MOD',
    'RULE_INVERSE_MOD', 'RULE_ITEM_CLAUSE', 'RULE_MEGA_RAYQUAZA_CLAUSE',
    'RULE_MOODY_CLAUSE', 'RULE_NFE_CLAUSE', 'RULE_OHKO_CLAUSE',
    'RULE_SAME_TYPE_CLAUSE', 'RULE_SLEEP_CLAUSE_MOD', 'RULE_SPECIES_CLAUSE',
    'RULE_SWAGGER_CLAUSE', 'RULE_SWITCH_PRIORITY_CLAUSE_MOD',
]) + [('z-move clause', 'RULE_ZMOVE_CLAUSE')]

SERVER_REPLY_HEADERS = trie_from_strings('ServerReply::', None, [
    'p1', 'p2', 'p3', 'p4',
    'gametype', 'gen', 'player', 'rule', 'sideupdate', 'split', 'start', 'teamsize', 'update',
    'clearpoke', 'poke', 'teampreview',
    'drag', 'turn'
]) + [('switch', 'ServerReply::switch_position')]

STAT_NAMES = [
    ('hp', 0),
    ('attack', 1), ('atk', 1),
    ('defense', 2), ('def', 2),
    ('specialattack', 3), ('spa', 3),
    ('specialdefense', 4), ('spd', 4),
    ('speed', 5), ('spe', 5),
]

TYPE_NAMES = trie_from_strings('Type::', lambda s: s.lower(), [
    'BUG', 'DRAGON', 'ELECTRIC', 'FIGHTING', 'FIRE', 'FLYING', 'GHOST',
    'GRASS', 'GROUND', 'ICE', 'NORMAL', 'POISON', 'PSYCHIC', 'ROCK',
    'WATER', 'DARK', 'STEEL', 'FAIRY'
])


TRANSLATE_POSITION_DOUBLES_P1 = [
    ('p1a', 4 + 1), ('p1b', 4 + 2),
    ('p2b', 4 - 1), ('p2a', 4 - 2),
]

TRANSLATE_POSITION_DOUBLES_P2 = [
    ('p2a', 4 + 1), ('p2b', 4 + 2),
    ('p1b', 4 - 1), ('p1a', 4 - 2),
]

TRANSLATE_POSITION_TRIPLES_P1 = [
    ('p1a', 4 + 1), ('p1b', 4 + 2), ('p1c', 4 + 3),
    ('p2c', 4 - 1), ('p2b', 4 - 2), ('p2a', 4 - 3),
]

TRANSLATE_POSITION_TRIPLES_P2 = [
    ('p2a', 4 + 1), ('p2b', 4 + 2), ('p2c', 4 + 3),
    ('p1c', 4 - 1), ('p1b', 4 - 2), ('p1a', 4 - 3),
]

TRANSLATE_POSITION_MULTI_P1 = [
    ('p1a', 4 + 1), ('p3b', 4 + 2),
    ('p4b', 4 - 1), ('p2a', 4 - 2),
]

TRANSLATE_POSITION_MULTI_P3 = [
    ('p3b', 4 + 1), ('p1a', 4 + 2),
    ('p2a', 4 - 1), ('p4b', 4 - 2),
]

TRANSLATE_POSITION_MULTI_P2 = [
    ('p2a', 4 + 1), ('p4b', 4 + 2),
    ('p3b', 4 - 1), ('p1a', 4 - 2),
]

TRANSLATE_POSITION_MULTI_P4 = [
    ('p4b', 4 + 1), ('p2a', 4 + 2),
    ('p1a', 4 - 1), ('p3b', 4 - 2),
]


from collections import namedtuple

Jump = namedtuple('Jump', ('char', 'to'))

class Node:
    __slots__ = ('jumps', 'jump_map', 'fixup')

    def __init__(self):
        self.jumps = []
        self.jump_map = {}
        self.fixup = None

    def add_entry(self, key, result, prev_key=''):
        if not key:
            if 0 in self.jump_map:
                raise ValueError('Repeated key: {}'.format(prev_key))
            jump = Jump(0, result)
            self.jumps.append(jump)
            self.jump_map[0] = jump
        else:
            k = ord(key[0])
            try:
                subnode = self.jump_map[k].to
            except KeyError:
                subnode = Node()
                jump = Jump(k, subnode)
                self.jumps.append(jump)
                self.jump_map[k] = jump
            subnode.add_entry(key[1:], result, prev_key + key[0])

    def compile(self, compilation, stack):
        def compile_part(first, last, fixup):
            distance = last - first
            assert distance >= 0
            if distance == 0:
                current = first
            elif distance % 2:
                # EVEN (sic) number of nodes
                ind1 = first + distance // 2
                n1 = self.jumps[ind1]
                n2 = self.jumps[ind1 + 1]
                if isinstance(n1.to, Node):
                    if isinstance(n2.to, Node):
                        if(len(n1.to.jumps) > len(n2.to.jumps)):
                            current = ind1
                        else:
                            current = ind1 + 1
                    else:
                        current = ind1
                else:
                    current = ind1 + 1
            else:
                # ODD (sic) number of nodes
                current = first + distance // 2
            
            if(fixup is not None):
                compilation[fixup] = len(compilation) // 4

            # Compile current
            current_jump = self.jumps[current]
            if len(compilation) >= 0xFFFF - 4:
                raise RuntimeError('Compilation too large')
            compilation.extend([current_jump.char, None, None, None])
            lencomp = len(compilation)
            
            if isinstance(current_jump.to, Node):
                current_jump.to.fixup = lencomp - 1
                stack.append(current_jump.to)
            else:
                compilation[lencomp - 1] = 'static_cast<uint_fast16_t>(' + str(current_jump.to) + ')'

            # Compile previous
            if current == first:
                compilation[lencomp - 3] = 0
            else:
                assert current > first
                compile_part(first, current - 1, lencomp - 3)
            
            # Compile next
            if current == last:
                compilation[lencomp - 2] = 0
            else:
                assert current < last
                compile_part(current + 1, last, lencomp - 2)
        
        if self.jump_map is None:
            raise RuntimeError('Repeated attempt to compile a Node')
        self.jump_map = None
        self.jumps.sort(key = lambda x: x.char)
        compile_part(0, len(self.jumps) - 1, self.fixup)


def compile_trie(name, source):
    keys = set()
    for k, v in source:
        if k in keys:
            raise ValueError('Repeated key in {}: {}'.format(name, k))
        if not isinstance(v, (int, str)):
            raise ValueError('Invalid value in {}: {}'.format(name, v))
        keys.add(k)
    source.sort(key=lambda x: x[0])

    start = Node()
    for k, v in source:
        start.add_entry(k, v)
    
    compilation = []
    stack = [start]
    while stack:
        next_node = stack.pop()
        next_node.compile(compilation, stack)

    for i in range(0, len(compilation), 4):
        for k in range(1, 3):
            assert isinstance(compilation[i + k], int)
            assert compilation[i + k] < len(compilation) // 4
        if isinstance(compilation[i + 3], int):
            assert isinstance(compilation[i + 3], int)
            assert compilation[i + 3] < len(compilation) // 4
        else:
            assert(isinstance(compilation[i + 3], str))

    for i in range(0, len(compilation), 4):
        compilation[i] = '\n\t' + str(compilation[i])
        for k in range(1, 4):
            compilation[i + k] = str(compilation[i + k])
    
    return """const uint_fast16_t {}[] {{{}
}};""".format(name, ', '.join(compilation))


def main():
    print(
"""// GENERATED by generate_tries.py
// See generate_tries.py for licensing info

#ifndef SEIMEI_LIBPSHOWDOWN_HPP_
#include <libPShowdown.hpp>
#endif

#include <cstdint>

//@formatter:off
namespace LIBPSHOWDOWN_NAMESPACE {
namespace {
""")
    for tn in LIST_OF_TRIES:
        print(compile_trie(tn, globals()[tn]))
        print()
    print("""}
}
//@formatter:on""")
    
if __name__ == '__main__':
    main()
