//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
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

#include "dll/util/timers.hpp" //Timers

#include "dataset.hpp" //names
#include "utils.hpp" //names

namespace lstm {

void train_global_lstm();
void prepare_encodings(const spot_dataset& dataset);
void prepare_ground_truth(const spot_dataset& dataset);
void prepare_keyword(const config& conf, const spot_dataset& dataset, const std::vector<std::string>& label);

//template<typename VT>
//double distance(const spot_dataset& dataset, const std::string& test_image, const VT& features){
    //auto& label = dataset.word_labelstktk

//}

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
void prepare_features(const config& conf, const std::string& folder_name, names test_image_names, const V1& test_features_a) {
    const std::string base_folder = ".lstm";
    const std::string folder = base_folder + "/data/";

    mkdir(base_folder.c_str(), 0777);
    mkdir(folder.c_str(), 0777);

    // Fucking kill me already
    if (conf.ak || conf.botany){
        const std::string sub_train_folder = folder + "/train/";
        const std::string sub_test_folder = folder + "/test/";

        mkdir(sub_train_folder.c_str(), 0777);
        mkdir(sub_test_folder.c_str(), 0777);
    }

    for(std::size_t t = 0; t < test_image_names.size(); ++t){
        auto& test_image = test_image_names[t];
        auto& test_features = test_features_a[t];

        // Local files
        const std::string file_path     = folder + "/" + test_image + ".htk";

        // Generate the feature file

        {
            std::ofstream os(file_path, std::ofstream::binary);

            htk_features_write(os, test_features);
        }
    }

    // Generate the list file

    const std::string lst_file_path = base_folder + "/" + folder_name + ".txt";
    std::ofstream os;
    os.open(lst_file_path, std::ios_base::app);

    for(std::size_t t = 0; t < test_image_names.size(); ++t){
        auto& test_image = test_image_names[t];

        os << test_image << "\n";
    }
}

template <typename V1>
void prepare_test_features(const config& conf, names test_image_names, const V1& test_features_a) {
    dll::auto_timer timer("lstm_test_features");

    // Just make sure we start with empty file
    {
        std::ofstream os;
        os.open(".lstm/test.txt", std::ios::out);
    }

    prepare_features(conf, "test", test_image_names, test_features_a);
}

inline void prepare_valid_features(const config& conf) {
    dll::auto_timer timer("lstm_valid_features");

    const std::string lst_file_path = ".lstm/valid.txt";

    {
        std::ofstream os;
        os.open(".lstm/valid.txt", std::ios::out);
    }
}

template <typename Functor>
void prepare_train_features(const config& conf, names train_image_names, Functor functor) {
    dll::auto_timer timer("lstm_train_features");

    static constexpr const std::size_t limit = 1000;

    std::vector<std::string> current_batch;
    current_batch.reserve(limit);

    // Just make sure we start with empty file
    {
        std::ofstream os;
        os.open(".lstm/train.txt", std::ios::out);
    }

    for(std::size_t i = 0; i < train_image_names.size();){
        current_batch.clear();

        auto end = std::min(i + limit, train_image_names.size());
        std::copy(train_image_names.begin() + i, train_image_names.begin() + end, std::back_inserter(current_batch));

        auto train_features_a = functor(current_batch);
        prepare_features(conf, "train", current_batch, train_features_a);

        i += (end - i);
    }
}

} //end of namespace hmm_mlpack
