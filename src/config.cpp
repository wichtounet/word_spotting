//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "cpp_utils/assert.hpp"
#include "config.hpp"

void print_usage() {
    std::cout << "Usage: spotter [options] <command> file [file...]" << std::endl;
    std::cout << "Supported commands: " << std::endl;
    std::cout << " * train" << std::endl;
    std::cout << " * features" << std::endl;
    std::cout << " * evaluate" << std::endl;
    std::cout << "Supported options: " << std::endl;
    std::cout << " -0 : Method 0 [Marti2001]" << std::endl;
    std::cout << " -1 : Method 1 (holistic)" << std::endl;
    std::cout << " -2 : Method 2 (patches)" << std::endl;
    std::cout << " -3 : Method 3 [Rath2007]" << std::endl;
    std::cout << " -4 : Method 4 [Rath2003]" << std::endl;
    std::cout << " -5 : Method 5 [Rodriguez2008]" << std::endl;
    std::cout << " -6 : Method 6 [Vinciarelli2004]" << std::endl;
    std::cout << " -7 : Method 7 [Terasawa2009]" << std::endl;
    std::cout << " -half : Takes half resolution images only" << std::endl;
    std::cout << " -quarter : Takes quarter resolution images only" << std::endl;
    std::cout << " -third : Takes third resolution images only" << std::endl;
    std::cout << " -svm : Use a SVM" << std::endl;
    std::cout << " -view : Load the DBN and visualize its weights" << std::endl;
    std::cout << " -sub : Takes only a subset of the dataset to train (holistic/patches only)" << std::endl;
    std::cout << " -fix : Don't optimize sc_band" << std::endl;
    std::cout << " -notrain : No evaluation on the training set" << std::endl;
    std::cout << " -novalid : No evaluation on the validation set" << std::endl;
    std::cout << " -washington : The dataset is Washington [default]" << std::endl;
    std::cout << " -parzival : The dataset is Parzival" << std::endl;
    std::cout << " -iam : The dataset is IAM" << std::endl;
    std::cout << " -hmm : Use HMM (with mlpack) in place of DTW" << std::endl;
    std::cout << " -htk : Use HTK in place of mlpack" << std::endl;
}

config parse_args(int argc, char** argv) {
    config conf;

    for (std::size_t i = 1; i < static_cast<size_t>(argc); ++i) {
        conf.args.emplace_back(argv[i]);
    }

    std::size_t i = 0;
    for (; i < conf.args.size(); ++i) {
        if (conf.args[i] == "-0") {
            conf.method = Method::Marti2001;
        } else if (conf.args[i] == "-1") {
            conf.method = Method::Holistic;
        } else if (conf.args[i] == "-2") {
            conf.method = Method::Patches;
        } else if (conf.args[i] == "-3") {
            conf.method = Method::Rath2007;
        } else if (conf.args[i] == "-4") {
            conf.method = Method::Rath2003;
        } else if (conf.args[i] == "-5") {
            conf.method = Method::Rodriguez2008;
        } else if (conf.args[i] == "-6") {
            conf.method = Method::Vinciarelli2004;
        } else if (conf.args[i] == "-7") {
            conf.method = Method::Terasawa2009;
        } else if (conf.args[i] == "-full") {
            //Simply here for consistency sake
        } else if (conf.args[i] == "-half") {
            conf.half = true;
        } else if (conf.args[i] == "-quarter") {
            conf.quarter = true;
        } else if (conf.args[i] == "-third") {
            conf.third = true;
        } else if (conf.args[i] == "-svm") {
            conf.svm = true;
        } else if (conf.args[i] == "-view") {
            conf.view = true;
        } else if (conf.args[i] == "-load") {
            conf.load = true;
        } else if (conf.args[i] == "-sub") {
            conf.sub = true;
        } else if (conf.args[i] == "-fix") {
            conf.fix = true;
        } else if (conf.args[i] == "-all") {
            conf.all = true;
        } else if (conf.args[i] == "-notrain") {
            conf.notrain = true;
        } else if (conf.args[i] == "-novalid") {
            conf.novalid = true;
        } else if (conf.args[i] == "-hmm") {
            conf.hmm = true;
        } else if (conf.args[i] == "-htk") {
            conf.htk = true;
        } else if (conf.args[i] == "-washington") {
            conf.washington = true;
            conf.parzival   = false;
            conf.iam        = false;
        } else if (conf.args[i] == "-parzival") {
            conf.washington = false;
            conf.parzival   = true;
            conf.iam        = false;
        } else if (conf.args[i] == "-iam") {
            conf.washington = false;
            conf.parzival   = false;
            conf.iam        = true;
        } else {
            break;
        }
    }

    conf.command = conf.args[i++];

    for (; i < conf.args.size(); ++i) {
        conf.files.push_back(conf.args[i]);
    }

    return conf;
}

std::string method_to_string(Method method){
    switch (method) {
        case Method::Marti2001:
            return "Marti2001";
        case Method::Holistic:
            return "Holistic";
        case Method::Patches:
            return "Patches";
        case Method::Rath2007:
            return "Rath2007";
        case Method::Rath2003:
            return "Rath2003";
        case Method::Rodriguez2008:
            return "Rodriguez2008";
        case Method::Vinciarelli2004:
            return "Vinciarelli2004";
        case Method::Terasawa2009:
            return "Terasawa2009";
    }

    cpp_unreachable("Unhandled method");

    return "invalid_method";
}
