#ifndef __CLASSIFY_IMAGE_H__
#define __CLASSIFY_IMAGE_H__

#include <string>

namespace classify {

std::string classifyImage(
    std::string model_path,
    std::string image_path,
    std::string labels_path,
    float input_mean,
    float input_std
    );

}

#endif  /*__CLASSIFY_IMAGE_H__*/
