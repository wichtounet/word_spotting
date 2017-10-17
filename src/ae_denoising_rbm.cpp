//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef SPOTTER_NO_AE

#include "dll/rbm/rbm.hpp"
#include "dll/dbn.hpp"

#include "ae_config.hpp" // Must be first

#include "ae_rbm.hpp"

#include "ae_evaluation.hpp"

namespace {

template<size_t Noise>
void denoising_rbm_evaluate(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& clean, float learning_rate, size_t epochs) {
    using network_t = typename dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<
                patch_height * patch_width, 50,
                dll::batch_size<batch_size>,
                dll::weight_decay<dll::decay_type::L2>,
                dll::momentum
        >::layer_t
    >, dll::noise<Noise>>::dbn_t;

    auto net = std::make_unique<network_t>();

    net->display();

    // Configure the network
    net->template layer_get<0>().learning_rate    = learning_rate;
    net->template layer_get<0>().initial_momentum = 0.9;
    net->template layer_get<0>().momentum         = 0.9;

    // Train as RBM
    net->pretrain_denoising(clean, epochs);

    auto folder = spot::evaluate_patches_ae<0, image_t>(dataset, set, conf, *net, train_word_names, test_image_names, false, params);
    std::cout << "AE-Result: Denoising-RBM(" << Noise << "):" << folder << std::endl;
}

} // end of anonymous namespace

void denoising_rbm_evaluate_all(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names test_image_names, parameters params, const std::vector<image_t>& clean){
    if (conf.denoising && conf.rbm) {
        auto lr = 1e-3;

        denoising_rbm_evaluate<0>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_rbm_evaluate<5>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_rbm_evaluate<10>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_rbm_evaluate<15>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_rbm_evaluate<20>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_rbm_evaluate<25>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_rbm_evaluate<30>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_rbm_evaluate<35>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_rbm_evaluate<40>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_rbm_evaluate<45>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
        denoising_rbm_evaluate<50>(dataset, set, conf, train_word_names, test_image_names, params, clean, lr, epochs);
    }
}

#endif // SPOTTER_NO_AE
