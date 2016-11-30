//=======================================================================
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_CONFIG_THIRD_HPP
#define WORD_SPOTTER_CONFIG_THIRD_HPP

namespace third {

//#define THIRD_RBM_1       //One layer of RBM
//#define THIRD_RBM_2       //Two layers of RBM
//#define THIRD_RBM_3       //Three layers of RBM

//#define THIRD_CRBM_PMP_1    //One layer of CRBM with Probabilistic Max Pooling (C1)
//#define THIRD_CRBM_PMP_2    //Two layers of CRBM with Probabilistic Max Pooling (C1/C2)
//#define THIRD_CRBM_PMP_3    //Three layers of CRBM with Probabilistic Max Pooling (C1/C2/C3)

//#define THIRD_CRBM_MP_1  //One layers of CRBM with Max Pooling after each layer (C1)
#define THIRD_CRBM_MP_2  //Two layers of CRBM with Max Pooling after each layer (C1/C2)
//#define THIRD_CRBM_MP_3  //Three layers of CRBM with Max Pooling after each layer (C1/C2/C3)
//#define THIRD_PATCH_CRBM_MP_2  //Patches -> Two layers of CRBM with Max Pooling after each layer (C1/C2)

//#define THIRD_COMPLEX_2 //Architecture to play around LCN
//#define THIRD_MODERN    //Architecture to play around

constexpr const std::size_t patch_height = 40; //Should not be changed
constexpr const std::size_t patch_width  = 20;

// Data augmentation
constexpr const std::size_t elastic_augment = 1; //not ready

constexpr const std::size_t epochs       = 10;
constexpr const std::size_t train_stride = 1;
constexpr const std::size_t test_stride  = 1;

constexpr const std::size_t NF1          = 9;
constexpr const std::size_t K1           = 12;
constexpr const std::size_t C1           = 2;
constexpr const std::size_t B1           = 128;
constexpr const dll::unit_type HT1       = dll::unit_type::RELU;
constexpr const dll::decay_type DT1      = dll::decay_type::L2;
constexpr const dll::sparsity_method SM1 = dll::sparsity_method::NONE;
constexpr const bool shuffle_1           = false;
constexpr const bool clipping_1          = true;

constexpr const std::size_t NF2          = 3;
constexpr const std::size_t K2           = 12;
constexpr const std::size_t C2           = 2;
constexpr const std::size_t B2           = 128;
constexpr const dll::unit_type HT2       = dll::unit_type::RELU;
constexpr const dll::decay_type DT2      = dll::decay_type::L2;
constexpr const dll::sparsity_method SM2 = dll::sparsity_method::NONE;
constexpr const bool shuffle_2           = false;
constexpr const bool clipping_2          = true;

constexpr const std::size_t NF3          = 3;
constexpr const std::size_t K3           = 48;
constexpr const std::size_t C3           = 2;
constexpr const std::size_t B3           = 64;
constexpr const dll::unit_type HT3       = dll::unit_type::BINARY;
constexpr const dll::decay_type DT3      = dll::decay_type::L2;
constexpr const dll::sparsity_method SM3 = dll::sparsity_method::NONE;
constexpr const bool shuffle_3           = true;
constexpr const bool clipping_3          = false;

const auto rate_0 = [](weight& value) { value = 0.1 * value; };
const auto rate_1 = [](weight& value) { value = 0.001 * value; };
const auto rate_2 = [](weight& value) { value = 1.0 * value; };

const auto momentum_0 = [](weight& ini, weight& fin) { ini = 0.9; fin = 0.9; };
const auto momentum_1 = [](weight& ini, weight& fin) { ini = 0.9; fin = 0.9; };
const auto momentum_2 = [](weight& ini, weight& fin) { ini = 1.0 * ini; fin = 1.0 * fin; };

const auto wd_l1_0 = [](weight& value) { value = 1.0 * value; };
const auto wd_l1_1 = [](weight& value) { value = 1.0 * value; };
const auto wd_l1_2 = [](weight& value) { value = 1.0 * value; };

const auto wd_l2_0 = [](weight& value) { value = 1.0 * value; };
const auto wd_l2_1 = [](weight& value) { value = 10.0 * value; };
const auto wd_l2_2 = [](weight& value) { value = 1.0 * value; };

const auto pbias_0 = [](weight& value) { value = 1.0 * value; };
const auto pbias_1 = [](weight& value) { value = 1.0 * value; };
const auto pbias_2 = [](weight& value) { value = 1.0 * value; };

#if defined(THIRD_CRBM_MP_2) || defined(THIRD_PATCH_CRBM_MP_2)
const auto pbias_lambda_0 = [](weight& value) { value = 10.0 * value; };
const auto pbias_lambda_1 = [](weight& value) { value = 10.0 * value; };
const auto pbias_lambda_2 = [](weight& value) { value = 1.0 * value; };
#else
const auto pbias_lambda_0 = [](weight& value) { value = 1.0 * value; };
const auto pbias_lambda_1 = [](weight& value) { value = 1.0 * value; };
const auto pbias_lambda_2 = [](weight& value) { value = 1.0 * value; };
#endif

const auto sparsity_target_0 = [](weight& value) { value = 10.0 * value; };
const auto sparsity_target_1 = [](weight& value) { value = 10.0 * value; };
const auto sparsity_target_2 = [](weight& value) { value = 1.0 * value; };

const auto clip_norm_1 = [](weight& t) { t = 5.0; };
const auto clip_norm_2 = [](weight& t) { t = 5.0; };
const auto clip_norm_3 = [](weight& t) { t = 5.0; };

} // end of namespace third

#endif
