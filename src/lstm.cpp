//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <map>

#include "etl/etl.hpp"

#include "config.hpp"
#include "lstm.hpp"
#include "grid.hpp"

// This is pretty impossible to reproduce
// 1) The code is not free
// 2) The code is bad
// 3) The code needs modification
const std::string bin_main_better   = "~/dev/taina_lstm/train/software/main_better.py";
const std::string bin_freeze_better = "~/dev/taina_lstm/train/software/freeze_better.py";
const std::string bin_hwrec         = "~/dev/tensorflow14/bazel-bin/tensorflow/examples/hwrec/main";

void lstm::prepare_keyword(const config& conf, const spot_dataset& dataset, const std::vector<std::string>& label){
    std::string label_name;

    for(auto& c : label){
        label_name += c;
    }

    const std::string base_folder     = ".lstm";
    const std::string label_folder = base_folder + "/" + label_name;

    mkdir(label_folder.c_str(), 0777);

    {
        std::ofstream os(".lstm/testparams.txt");
        //fileSpelling      tensorflow/examples/hwrec/data/lm/iamWC2gram.spl
        //fileLanguageModel tensorflow/examples/hwrec/data/lm/iamWC2gram.txt

        os << "testids .lstm/test.txt \n";
        //os << "groundtruth .lstm/gt.txt \n";
        os << "fileEncodings .lstm/encodings.txt \n";
        os << "fileNetwork .lstm/graph/graph_frozen_this_code_sucks.pbtxt \n";
        os << "dirFeatures .lstm/data/\n";
        os << "verbose 0\n";
        os << "gsf 1.0 \n";
        os << "wip 0.0 \n";
        os << "obsLogbase logNat \n";
        os << "compression 0.999 \n";
        os << "nodeNbest 4000 \n";
        os << "wordNbest 10 \n";
        os << "sentencebest 1 \n";
        os << "fileSpelling .lstm/" << label_name << "/word_spelling\n";
        //os << "fileLanguageModel .lstm/" << label_name << "/lm\n";
        //os << "dirEvaluation .lstm/" << label_name << "/eval/\n";
        os << "dirRecognition .lstm/" << label_name << "/rec/\n";
        os << "postfixObservation txt \n";
        os << "postfixRecognition rec \n";
        os << "wordBased 0 \n";
    }

    {
        std::ofstream os(".lstm/" + label_name + "/word_spelling");

        std::string before = "";
        for(auto& c : label){
            os << before << c;
            before = "-";
        }
        os << before << "sp";
        os << "\t";
        for(auto& c : label){
            os << " " << c;
        }
        os << " " << "sp";
        os << "\n";
    }

    {
        //TODO Should we use the total list of characters or only label ?

        std::ofstream os(".lstm/" + label_name + "/char_spelling");

        for(auto& c : label){
            os << c << "\t" << c << "\n";
        }
    }

    {
        //std::ofstream os(".lstm/" + label_name + "/lm");

        //bigram
    }

    {
        std::string command = bin_hwrec;

        auto result = exec_command(command);

        //if (result.first) {
            std::cout << "hwrec failed with result code: " << result.first << std::endl;
            std::cout << "Command: " << command << std::endl;
            std::cout << result.second << std::endl;
        //}

    }
}

void lstm::train_global_lstm() {
    const std::string base_folder     = ".lstm";
    const std::string training_folder = base_folder + "/training/";
    const std::string graph_folder    = base_folder + "/graph/";

    mkdir(training_folder.c_str(), 0777);
    mkdir(graph_folder.c_str(), 0777);

    // 1. Train the model

    {
        std::string command = "python " + bin_main_better;

        auto result = exec_command(command);

        if (result.first) {
            std::cout << "python train script failed with result code: " << result.first << std::endl;
            std::cout << "Command: " << command << std::endl;
            std::cout << result.second << std::endl;
        }
    }

    // 2. Freeze the model

    {
        std::string command = "python " + bin_freeze_better;

        auto result = exec_command(command);

        if (result.first) {
            std::cout << "python freeze script failed with result code: " << result.first << std::endl;
            std::cout << "Command: " << command << std::endl;
            std::cout << result.second << std::endl;
        }
    }
}

void lstm::prepare_ground_truth(const spot_dataset& dataset){
    std::ofstream os(".lstm/gt.txt");

    for (auto& label : dataset.word_labels) {
        os << label.first << " ";

        std::string before;
        for(auto& c : label.second){
            os << before << c;
            before = "-";
        }

        os << "\n";
    }
}

void lstm::prepare_encodings(const spot_dataset& dataset){
    std::map<std::string, size_t> encodings;

    std::unordered_map<std::string, std::vector<std::string>> word_labels;

    for (auto& label : dataset.word_labels) {
        for (auto& c : label.second) {
            if(!encodings.count(c)){
                encodings[c] = encodings.size();
            }
        }
    }

    std::vector<std::string> sorted_encodings(encodings.size());

    for(auto& value : encodings){
        sorted_encodings[value.second - 1] = value.first;
    }

    std::ofstream os(".lstm/encodings.txt");

    for(size_t i = 0; i < sorted_encodings.size(); ++i){
        os << sorted_encodings[i] << " " << i + 1 << '\n';
    }

    // SHOULD NOT BE NECESSARY! FUCK
    os << "sp " << sorted_encodings.size() + 1 << '\n';
}
