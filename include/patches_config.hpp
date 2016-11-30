//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
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

#if defined(THIRD_CRBM_PMP_2) || defined(THIRD_CRBM_MP_2) || defined(THIRD_RBM_2) || defined(THIRD_COMPLEX_2) || defined(THIRD_MODERN)
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

#define silence_l2_warnings() \
    cpp_unused(K2);           \
    cpp_unused(C2);           \
    cpp_unused(L2);           \
    cpp_unused(NH2_1);        \
    cpp_unused(NH2_2);        \
    cpp_unused(clipping_2);   \
    cpp_unused(shuffle_2);

#define silence_l3_warnings() \
    cpp_unused(K3);           \
    cpp_unused(C3);           \
    cpp_unused(L3);           \
    cpp_unused(NH3_1);        \
    cpp_unused(NH3_2);        \
    cpp_unused(clipping_3);   \
    cpp_unused(shuffle_3);

#define copy_from_namespace(ns)                                       \
    static constexpr const std::size_t K1         = ns::K1;           \
    static constexpr const std::size_t C1         = ns::C1;           \
    static constexpr const std::size_t NF1        = ns::NF1;          \
    static constexpr const std::size_t NV1_1      = ns::patch_height; \
    static constexpr const std::size_t NV1_2      = ns::patch_width;  \
    static constexpr const std::size_t NH1_1      = NV1_1 - NF1 + 1;  \
    static constexpr const std::size_t NH1_2      = NV1_2 - NF1 + 1;  \
    static constexpr const std::size_t shuffle_1  = ns::shuffle_1;    \
    static constexpr const std::size_t clipping_1 = ns::clipping_1;   \
    static constexpr const std::size_t K2         = ns::K2;           \
    static constexpr const std::size_t C2         = ns::C2;           \
    static constexpr const std::size_t NF2        = ns::NF2;          \
    static constexpr const std::size_t NV2_1      = NH1_1 / C1;       \
    static constexpr const std::size_t NV2_2      = NH1_2 / C1;       \
    static constexpr const std::size_t NH2_1      = NV2_1 - NF2 + 1;  \
    static constexpr const std::size_t NH2_2      = NV2_2 - NF2 + 1;  \
    static constexpr const std::size_t shuffle_2  = ns::shuffle_2;    \
    static constexpr const std::size_t clipping_2 = ns::clipping_2;   \
    static constexpr const std::size_t K3         = ns::K3;           \
    static constexpr const std::size_t C3         = ns::C3;           \
    static constexpr const std::size_t NF3        = ns::NF3;          \
    static constexpr const std::size_t NV3_1      = NH2_1 / C2;       \
    static constexpr const std::size_t NV3_2      = NH2_2 / C2;       \
    static constexpr const std::size_t NH3_1      = NV3_1 - NF3 + 1;  \
    static constexpr const std::size_t NH3_2      = NV3_2 - NF3 + 1;  \
    static constexpr const std::size_t shuffle_3  = ns::shuffle_3;    \
    static constexpr const std::size_t clipping_3 = ns::clipping_3;
