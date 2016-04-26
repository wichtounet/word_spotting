//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#define FULL_DEBUG

#include <random>
#include <set>
#include <sstream>
#include <fstream>

#include <sys/stat.h> //mkdir

#include "cpp_utils/io.hpp" //Binary write

#include "dll/dbn.hpp" //Timers

#include "dataset.hpp" //names
#include "utils.hpp" //names

namespace hmm_htk {

using hmm_p = std::string;

using thread_pool = cpp::default_thread_pool<>;

std::size_t select_gaussians(const config& conf);
void write_log(const std::string& result, const std::string& file);

hmm_p train_global_hmm(const config& conf, const spot_dataset& dataset, names train_word_names);
hmm_p prepare_test_keywords(const spot_dataset& dataset, names training_images);

void global_likelihood_all(const config& conf, thread_pool& pool, const hmm_p& base_folder, names test_image_names, std::vector<double>& global_likelihoods);
void global_likelihood_many(const config& conf, const hmm_p& base_folder, names test_image_names, std::vector<double>& global_likelihoods, std::size_t t, std::size_t start, std::size_t end);

void keyword_likelihood_all(const config& conf, thread_pool& pool, const hmm_p& base_folder, const hmm_p& folder, names test_image_names, std::vector<double>& keyword_likelihoods);
void keyword_likelihood_many(const config& conf, const hmm_p& base_folder, const hmm_p& folder, names test_image_names, std::vector<double>& keyword_likelihoods, std::size_t t, std::size_t start);

double hmm_distance(const spot_dataset& dataset, const std::string& test_image, names training_images, double global_acc, double keyword_acc);

template <typename V1>
void htk_features_write(std::ostream& os, const V1& test_features){
    const auto n_features = test_features[0].size();

    cpp::binary_write(os, static_cast<int32_t>(test_features.size()));       //Number of observations
    cpp::binary_write(os, static_cast<int32_t>(1));                          //Dummy HTK_SAMPLE_RATE
    cpp::binary_write(os, static_cast<int16_t>(n_features * sizeof(float))); //Observation size
    cpp::binary_write(os, static_cast<int16_t>(9));                          //Used defined sample kind = 9

    //Write all the values
    for (decltype(auto) feature_vector : test_features) {
        for (auto feature : feature_vector) {
            cpp::binary_write(os, static_cast<float>(feature));
        }
    }
}

template <typename V1>
void prepare_features(const std::string& folder_name, names test_image_names, const V1& test_features_a, bool lst_file) {
    const std::string base_folder = ".hmm";
    const std::string folder = base_folder + "/" + folder_name;

    mkdir(base_folder.c_str(), 0777);
    mkdir(folder.c_str(), 0777);

    for(std::size_t t = 0; t < test_image_names.size(); ++t){
        auto& test_image = test_image_names[t];
        auto& test_features = test_features_a[t];

        // Local files
        const std::string features_file = folder + "/" + test_image + ".lst";
        const std::string file_path     = folder + "/" + test_image + ".htk";

        // Generate the file with the list of feature files

        if(lst_file) {
            std::ofstream os(features_file);
            os << file_path << "\n";
        }

        // Generate the feature file

        {
            std::ofstream os(file_path, std::ofstream::binary);

            htk_features_write(os, test_features);
        }
    }
}

template <typename V1>
void prepare_test_features(names test_image_names, const V1& test_features_a) {
    dll::auto_timer timer("htk_test_features");
    prepare_features("test", test_image_names, test_features_a, false);
}

template <typename Functor>
void prepare_train_features(names train_image_names, Functor functor) {
    dll::auto_timer timer("htk_train_features");

    static constexpr const std::size_t limit = 1000;

    std::vector<std::string> current_batch;
    current_batch.reserve(limit);

    for(std::size_t i = 0; i < train_image_names.size();){
        current_batch.clear();

        auto end = std::min(i + limit, train_image_names.size());
        std::copy(train_image_names.begin() + i, train_image_names.begin() + end, std::back_inserter(current_batch));

        auto train_features_a = functor(current_batch);
        prepare_features("train", current_batch, train_features_a, false);

        i += (end - i);
    }
}

} //end of namespace hmm_mlpack
