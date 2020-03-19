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

#ifndef SEIMEI_COMMON_HPP_
#define SEIMEI_COMMON_HPP_

#include <cstdlib>
#include <stdexcept>
#include <string>

#ifdef SEIMEI_DOUBLE_PRECISION
#define FLOAT double
#define FLOAT_TEXT "double"
#else
#define FLOAT float
#define FLOAT_TEXT "float"
#endif

#define FCAST(x) static_cast<FLOAT>(x)
#define F1 static_cast<FLOAT>(1)

namespace seimei {
namespace {
[[noreturn]] inline void unreachable(const std::string& where) {
	throw std::logic_error(where + std::string(": INTERNAL ERROR: unreachable code reached"));
}
}
}

#endif
