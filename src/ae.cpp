//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <string>

#include "cpp_utils/parallel.hpp"

#include "ae.hpp"
#include "ae_config.hpp"

// Dense versions
#include "ae_dense.hpp"
#include "ae_rbm.hpp"

// Conv versions
#include "ae_crbm.hpp"
#include "ae_dense.hpp"
#include "ae_conv.hpp"
#include "ae_crbm.hpp"

#include "config.hpp"
#include "standard.hpp"
#include "utils.hpp"
#include "reports.hpp"
#include "features.hpp"   //Features exporting
#include "evaluation.hpp" //evaluation utilities

namespace {

void log_scaling(){
#ifdef LOCAL_FRAME_NORMALIZATION
    std::cout << "Local Frame Normalization" << std::endl;
#endif

#ifdef LOCAL_L2_NORMALIZATION
    std::cout << "Local L2 Normalization" << std::endl;
#endif

#ifdef GLOBAL_FRAME_NORMALIZATION
    std::cout << "Global Frame Normalization" << std::endl;
#endif

#ifdef GLOBAL_L2_NORMALIZATION
    std::cout << "Global L2 Normalization" << std::endl;
#endif

#ifdef LOCAL_LINEAR_SCALING
    std::cout << "Local Linear Scaling" << std::endl;
#endif

#ifdef LOCAL_MEAN_SCALING
    std::cout << "Local Mean Scaling" << std::endl;
#endif
}

} // end of anonymous namespace

void ae_train(const spot_dataset& dataset, const spot_dataset_set& set, config& conf, names train_word_names, names train_image_names, names test_image_names) {
    log_scaling();

    std::cout << "Use a third of the resolution" << std::endl;

    //Pass information to the next passes (evaluation)
    conf.patch_width  = patch_width;
    conf.train_stride = train_stride;
    conf.test_stride  = test_stride;

    //Train the DBN
    std::vector<image_t> training_patches;
    training_patches.reserve(train_image_names.size() * 10);

    std::cout << "Generate patches ..." << std::endl;

    for (auto& name : train_image_names) {
        // Insert the patches from the original image
        auto patches = mat_to_patches_t<image_t>(conf, dataset.word_images.at(name), true);
        std::copy(patches.begin(), patches.end(), std::back_inserter(training_patches));
    }

    std::cout << "... " << training_patches.size() << " patches extracted" << std::endl;

    std::cout << "Switch to optimal parameters" << std::endl;
    parameters params;
    params.sc_band = 0.05;
    std::cout << "\tsc_band: " << params.sc_band << std::endl;

    // Dense modules
    dense_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    rbm_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);

    // Deep Dense modules
    deep_rbm_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    deep_dense_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    stacked_dense_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);

    // Conv modules
    conv_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    crbm_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);

    // Conv+pooling modules
    conv_mp_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    crbm_mp_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    crbm_pmp_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);

    // Deep Conv modules
    deep_conv_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    stacked_conv_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    deep_crbm_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);

    // Deep Conv+pooling modules
    deep_conv_mp_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    stacked_conv_mp_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    deep_crbm_mp_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
    deep_crbm_pmp_evaluate_all(dataset, set, conf, train_word_names, test_image_names, params, training_patches);
}
