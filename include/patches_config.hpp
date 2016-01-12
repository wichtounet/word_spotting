//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/assert.hpp"

#include "dll/unit_type.hpp"
#include "dll/sparsity_method.hpp"
#include "dll/decay_type.hpp"

//The different configurations
#include "config_third.hpp"
#include "config_half.hpp"
#include "config_full.hpp"

#if defined(THIRD_CRBM_PMP_1) || defined(THIRD_CRBM_MP_1) || defined(THIRD_RBM_1)
#define THIRD_LEVELS 1
#endif

#if defined(THIRD_CRBM_PMP_2) || defined(THIRD_CRBM_MP_2) || defined(THIRD_RBM_2)
#define THIRD_LEVELS 2
#endif

#if defined(THIRD_CRBM_PMP_3) || defined(THIRD_CRBM_MP_3) || defined(THIRD_RBM_3)
#define THIRD_LEVELS 3
#endif

#if defined(HALF_CRBM_PMP_1) || defined(HALF_CRBM_MP_1)
#define HALF_LEVELS 1
#endif

#if defined(HALF_CRBM_PMP_2) || defined(HALF_CRBM_MP_2)
#define HALF_LEVELS 2
#endif

#if defined(HALF_CRBM_PMP_3) || defined(HALF_CRBM_MP_3)
#define HALF_LEVELS 3
#endif

#if defined(FULL_CRBM_PMP_1) || defined(FULL_CRBM_MP_1)
#define FULL_LEVELS 1
#endif

#if defined(FULL_CRBM_PMP_2) || defined(FULL_CRBM_MP_2)
#define FULL_LEVELS 2
#endif

#if defined(FULL_CRBM_PMP_3) || defined(FULL_CRBM_MP_3)
#define FULL_LEVELS 3
#endif

#if !defined(HALF_LEVELS) || !defined(THIRD_LEVELS) || !defined(FULL_LEVELS)
static_assert(false, "Invalid configuration");
#endif
