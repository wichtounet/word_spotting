//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/rbm/conv_rbm_mp.hpp"
#include "dll/dbn.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_crbm.hpp"

#include "ae_evaluation.hpp"

namespace {

template<size_t Noise>
void denoising_deep_crbm_pmp_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& clean, float learning_rate, size_t epochs) {
    static constexpr size_t K = 5;
    static constexpr size_t KK = 6;

    static constexpr size_t K1 = 9;
    static constexpr size_t K2 = 3;

    static constexpr size_t NH1_1 = patch_height - K1 + 1;
    static constexpr size_t NH1_2 = patch_width - K1 + 1;

    using network_t = typename dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_mp<
                1, patch_height, patch_width,
                KK, K1, K1, 2,
                dll::batch_size<batch_size>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::momentum
            >,
            dll::conv_rbm_mp<
                KK, NH1_1 / 2, NH1_2 / 2,
                K, K2, K2, 2,
                dll::batch_size<batch_size>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::momentum
            >
        >,
        dll::batch_mode, dll::noise<Noise>
    >::dbn_t;

    auto net = std::make_unique<network_t>();

    net->display();
    std::cout << net->output_size() << " features" << std::endl;

    // Configure the network
    net->template layer_get<0>().learning_rate    = 1e-6;
    net->template layer_get<0>().initial_momentum = 0.9;
    net->template layer_get<0>().momentum         = 0.9;

    net->template layer_get<1>().learning_rate    = learning_rate;
    net->template layer_get<1>().initial_momentum = 0.9;
    net->template layer_get<1>().momentum         = 0.9;

    // Train as RBM
    net->pretrain_denoising(clean, epochs);

    auto folder = spot::evaluate_patches_ae<1, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Denoising-Deep-CRBM-PMP(" << Noise << "):" << folder << std::endl;
}

} // end of anonymous namespace

void denoising_deep_crbm_pmp_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& clean){
    if (conf.denoising && conf.rbm) {
        auto lr = 1e-3;

        denoising_deep_crbm_pmp_evaluate<0>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_deep_crbm_pmp_evaluate<5>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_deep_crbm_pmp_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_deep_crbm_pmp_evaluate<15>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_deep_crbm_pmp_evaluate<20>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_deep_crbm_pmp_evaluate<25>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_deep_crbm_pmp_evaluate<30>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_deep_crbm_pmp_evaluate<35>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_deep_crbm_pmp_evaluate<40>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_deep_crbm_pmp_evaluate<45>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_deep_crbm_pmp_evaluate<50>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
    }
}
