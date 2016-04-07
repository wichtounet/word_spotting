//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "etl/etl.hpp"

#include "config.hpp"
#include "hmm_htk.hpp"

std::size_t hmm_htk::select_gaussians(const config& conf){
    if(conf.washington){
        return hmm_htk::n_hmm_gaussians_gw;
    } else if(conf.parzival){
        return hmm_htk::n_hmm_gaussians_par;
    } else if(conf.iam){
        return hmm_htk::n_hmm_gaussians_iam;
    } else {
        std::cout << "ERROR: Dataset is not handled in select_gaussians" << std::endl;
        return 0;
    }
}

void hmm_htk::write_log(const std::string& result, const std::string& file){
#ifdef WRITE_LOG
    std::ofstream os(file);
    os << result;
#else
    cpp_unused(result);
    cpp_unused(file);
#endif
}
