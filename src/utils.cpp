//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <fstream>

#include "utils.hpp"

etl::dyn_matrix<weight, 3> mat_for_patches(const config& /*conf*/, const cv::Mat& image){
    etl::dyn_matrix<weight, 3> training_image(std::size_t(1), std::size_t(image.size().height), std::size_t(image.size().width));

    for(std::size_t y = 0; y < static_cast<std::size_t>(image.size().height); ++y){
        for(std::size_t x = 0; x < static_cast<std::size_t>(image.size().width); ++x){
            auto pixel = image.at<uint8_t>(cv::Point(x, y));

            training_image(0, y, x) = pixel == 0 ? 0.0 : 1.0;

            if(pixel != 0 && pixel != 255){
                std::cout << "The input image is not binary! pixel:" << static_cast<int>(pixel) << std::endl;
            }
        }
    }

    return training_image;
}
