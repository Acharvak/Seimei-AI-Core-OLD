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

// A C++ wrapper for OpenBLAS or, potentially, other BLAS implementations.

#ifndef SEIMEI_BLAS_HPP_
#define SEIMEI_BLAS_HPP_

#include <cblas.h>
#include <cstdint>

namespace seimei {
namespace {
/**
 * BLAS GEMV for floats
 *
 * @param    matrix          A matrix
 * @param    rows            Number of rows in the matrix
 * @param    cols            Number of columns in the matrix
 * @param    transpose       Whether to transpose the matrix
 * @param    alpha           A scalar
 * @param    vector          A vector with M elements. If NOT transpose,
 *     M = cols, else M = rows
 * @param    beta            Another scalar
 * @param    result          A vector with N elements. If NOT transpose,
 *     N = rows, else N = cols
 * @param    matrix_is_cm    If true, the matrix is in column-major
 *     order, otherwise in row-major
 *
 * WARNING: all pointer parameters are assumed to be `restrict`
 *
 * When the call returns, if NOT transpose,
 *
 * * result = alpha * matrix @ vector + beta * result
 *
 * Else:
 *
 * * result = alpha * matrix transposed @ vector + beta * result
 *
 * where @ is matrix-vector multiplication
 */
inline void gemv(const float * matrix, size_t rows, size_t cols, bool transpose,
		float alpha, const float * vector, float beta, float * result,
		bool matrix_is_cm = false) {
	cblas_sgemv((matrix_is_cm ? CblasColMajor : CblasRowMajor),
			transpose ? CblasTrans : CblasNoTrans,
			static_cast<blasint>(rows), static_cast<blasint>(cols), alpha, matrix,
			(matrix_is_cm ? static_cast<blasint>(rows) : static_cast<blasint>(cols)),
			vector, 1, beta, result, 1);
}

/// BLAS GEMV for doubles
inline void gemv(const double * matrix, size_t rows, size_t cols, bool transpose,
		double alpha, const double * vector, double beta, double * result,
		bool matrix_is_cm = false) {
	cblas_dgemv((matrix_is_cm ? CblasColMajor : CblasRowMajor),
			transpose ? CblasTrans : CblasNoTrans,
			static_cast<blasint>(rows), static_cast<blasint>(cols), alpha, matrix,
			(matrix_is_cm ? static_cast<blasint>(rows) : static_cast<blasint>(cols)),
			vector, 1, beta, result, 1);
}

/**
 * BLAS AXPY for floats
 *
 * @param    alpha    A scalar
 * @param    X        A vector
 * @param    Y        A vector with the same size
 * @param    size     Number of elements in X and Y
 *
 * When the call returns:
 *
 * * Y = alpha * X + Y
 */
inline void axpy(float alpha, const float * X, float * Y, size_t size) {
	cblas_saxpy(static_cast<blasint>(size), alpha, X, 1, Y, 1);
}

/// BLAS AXPY for doubles
inline void axpy(double alpha, const double * X, double * Y, size_t size) {
	cblas_daxpy(static_cast<blasint>(size), alpha, X, 1, Y, 1);
}

/**
 * BLAS GER for floats.
 *
 * @param    alpha           A scalar
 * @param    matrix          A matrix
 * @param    rows            Number of rows in the matrix
 * @param    cols            Number of columns in the matrix
 * @param    X               A vector with `rows` elements
 * @param    Y               A vector with `cols` elements
 * @param    matrix_is_cm    If true, the matrix is in column-major
 *     order, otherwise in row-major
 *
 * When the call returns:
 *
 * * matrix = alpha * X @ Y transposed + matrix
 */
inline void ger(float alpha, float * matrix, size_t rows, size_t cols,
		const float * X, const float * Y, bool matrix_is_cm = false) {
	cblas_sger((matrix_is_cm ? CblasColMajor : CblasRowMajor),
			static_cast<blasint>(rows), static_cast<blasint>(cols),
			alpha, X, 1, Y, 1, matrix,
			(matrix_is_cm ? static_cast<blasint>(rows) : static_cast<blasint>(cols)));
}

/// BLAS GER for doubles
inline void ger(double alpha, double * matrix, size_t rows, size_t cols,
		const double * X, const double * Y, bool matrix_is_cm = false) {
	cblas_dger((matrix_is_cm ? CblasColMajor : CblasRowMajor),
			static_cast<blasint>(rows), static_cast<blasint>(cols),
			alpha, X, 1, Y, 1, matrix,
			(matrix_is_cm ? static_cast<blasint>(rows) : static_cast<blasint>(cols)));
}
}
}

#endif
