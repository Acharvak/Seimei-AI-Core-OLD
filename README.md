This is a planned neural-network based bot for [Pokémon Showdown!](https://pokemonshowdown.com/)

Doesn't quite play Pokémon yet. The neural network does pass generalized tests. Work is currently being done on handling the Showdown protocol (libPShowdown).

Also includes:

* Google Test Framework (in lib/googletest as a submodule)
* Niels Lohmann's JSON for Modern C++ (in lib/nlohmann-json as a copy)

Requires a C++17 compiler, CMake 3.12+, the [OpenBLAS library](https://www.openblas.net/) (other BLAS implementations may or may not work) and libhdf5 (1.10+).

License: Apache 2.0, see LICENSE.txt. Submodules and specific files may be licensed differently.
