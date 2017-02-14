//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_AE_CONFIG_HPP
#define WORD_SPOTTER_AE_CONFIG_HPP

#include "etl/etl.hpp" //For fast_dyn_matrix

//#define LOCAL_FRAME_NORMALIZATION
//#define LOCAL_L2_NORMALIZATION
//#define GLOBAL_FRAME_NORMALIZATION
#define GLOBAL_L2_NORMALIZATION

//#define LOCAL_LINEAR_SCALING
#define LOCAL_MEAN_SCALING

constexpr const auto patch_width  = 20;
constexpr const auto patch_height = 40;
constexpr size_t batch_size       = 128;
constexpr size_t epochs           = 10;
constexpr const auto train_stride = 1;
constexpr const auto test_stride  = 1;

using image_t = etl::fast_dyn_matrix<float, 1, patch_height, patch_width>;

#endif
