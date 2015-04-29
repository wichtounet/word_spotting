//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_CONFIG_THIRD_HPP
#define WORD_SPOTTER_CONFIG_THIRD_HPP

namespace third {

//#define CRBM_PMP_2  //Two layers of CRBM with Probabilistic Max Pooling
//#define CRBM_PMP_3  //Three layers of CRBM with Probabilistic Max Pooling
//#define CRBM_MP_2  //Two layers of CRBM with Max Pooling after each layer (C1/C2)
#define CRBM_MP_3  //Three layers of CRBM with Max Pooling after each layer (C1/C2/C3)

constexpr const std::size_t width = 220;        //Should not be changed
constexpr const std::size_t height = 40;        //Should not be changed

constexpr const std::size_t patch_width = 40;   //Should not be changed
constexpr const std::size_t patch_height = 40;  //Should not be changed

constexpr const std::size_t epochs = 25;
constexpr const std::size_t patch_stride = 20;

constexpr const std::size_t NF1 = 17;
constexpr const std::size_t K1 = 40;
constexpr const std::size_t C1 = 2;
constexpr const std::size_t B1 = 25;
constexpr const dll::unit_type HT1 = dll::unit_type::BINARY;
constexpr const dll::decay_type DT1 = dll::decay_type::L2;
constexpr const dll::sparsity_method SM1 = dll::sparsity_method::NONE;

constexpr const std::size_t NF2 = 5;
constexpr const std::size_t K2 = 40;
constexpr const std::size_t C2 = 2;
constexpr const std::size_t B2 = 25;
constexpr const dll::unit_type HT2 = dll::unit_type::BINARY;
constexpr const dll::decay_type DT2 = dll::decay_type::L2;
constexpr const dll::sparsity_method SM2 = dll::sparsity_method::NONE;

constexpr const std::size_t NF3 = 3;
constexpr const std::size_t K3 = 40;
constexpr const std::size_t C3 = 1;
constexpr const std::size_t B3 = 25;
constexpr const dll::unit_type HT3 = dll::unit_type::BINARY;
constexpr const dll::decay_type DT3 = dll::decay_type::L2;
constexpr const dll::sparsity_method SM3 = dll::sparsity_method::NONE;

const auto rate_0 = [](double& value){ value = 1.0 * value; };
const auto rate_1 = [](double& value){ value = 1.0 * value; };
const auto rate_2 = [](double& value){ value = 1.0 * value; };

const auto momentum_0 = [](double& ini, double& fin){ ini = 1.0 * ini; fin = 1.0 * fin;};
const auto momentum_1 = [](double& ini, double& fin){ ini = 1.0 * ini; fin = 1.0 * fin;};
const auto momentum_2 = [](double& ini, double& fin){ ini = 1.0 * ini; fin = 1.0 * fin;};

const auto wd_l1_0 = [](double& value){ value = 1.0 * value; };
const auto wd_l1_1 = [](double& value){ value = 1.0 * value; };
const auto wd_l1_2 = [](double& value){ value = 1.0 * value; };

const auto wd_l2_0 = [](double& value){ value = 1.0 * value; };
const auto wd_l2_1 = [](double& value){ value = 1.0 * value; };
const auto wd_l2_2 = [](double& value){ value = 1.0 * value; };

const auto pbias_0 = [](double& value){ value = 2.0 * value; };
const auto pbias_1 = [](double& value){ value = 2.0 * value; };
const auto pbias_2 = [](double& value){ value = 2.0 * value; };

const auto pbias_lambda_0 = [](double& value){ value = 1.0 * value; };
const auto pbias_lambda_1 = [](double& value){ value = 1.0 * value; };
const auto pbias_lambda_2 = [](double& value){ value = 1.0 * value; };

} // end of namespace third

#endif
