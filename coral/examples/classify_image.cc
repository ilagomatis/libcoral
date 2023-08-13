
#include "coral/examples/classify_image.h"

#include <cmath>
#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/classification/adapter.h"
#include "coral/examples/file_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"

namespace classify {

  std::string classifyImage(
      std::string model_path,
      std::string image_path,
      std::string labels_path,
      float input_mean,
      float input_std
      ){
    /*std::cout<<std::endl;
    std::cout<<"Model_path: "<<model_path<<std::endl;
    std::cout<<"Image_path: "<<image_path<<std::endl;
    std::cout<<"Labels_path: "<<labels_path<<std::endl;
    std::cout<<"input_mean: "<<input_mean<<std::endl;
    std::cout<<"input_std: "<<input_std<<std::endl;
    */
    // Load the model.
    const auto model = coral::LoadModelOrDie(model_path);

    auto edgetpu_context = coral::ContainsEdgeTpuCustomOp(*model)
                              ? coral::GetEdgeTpuContextOrDie()
                              : nullptr;
    auto interpreter =
        coral::MakeEdgeTpuInterpreterOrDie(*model, edgetpu_context.get());

    CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    CHECK_EQ(interpreter->inputs().size(), 1);
    const auto* input_tensor = interpreter->input_tensor(0);
    CHECK_EQ(input_tensor->type, kTfLiteUInt8)
        << "Only support uint8 input type.";
    const float scale = input_tensor->params.scale;
    const float zero_point = input_tensor->params.zero_point;
    const float mean = input_mean;
    const float std = input_std;
    auto input = coral::MutableTensorData<uint8_t>(*input_tensor);
    
    
    if (std::abs(scale * std - 1) < 1e-5 && std::abs(mean - zero_point) < 1e-5) {
      // Read the image directly into input tensor as there is no preprocessing
      // needed.
      std::cout << "Input data does not require preprocessing." << std::endl;
      coral::ReadFileToOrDie(image_path,
                            reinterpret_cast<char*>(input.data()), input.size());
    } else {
      std::cout << "Input data requires preprocessing." << std::endl;
      std::vector<uint8_t> image_data(input.size());
      coral::ReadFileToOrDie(image_path,
                            reinterpret_cast<char*>(image_data.data()),
                            input.size());
      for (int i = 0; i < input.size(); ++i) {
        const float tmp = (image_data[i] - mean) / (std * scale) + zero_point;
        if (tmp > 255) {
          input[i] = 255;
        } else if (tmp < 0) {
          input[i] = 0;
        } else {
          input[i] = static_cast<uint8_t>(tmp);
        }
      }
    }
    

    CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

    // Read the label file.
    auto labels = coral::ReadLabelFile(labels_path);

    std::string out = "";
  
    
    for (auto result : coral::GetClassificationResults(*interpreter, 0.0f, 3)) {
      out = out + "-------------------------" + "\n";
      out = out + labels[result.id];
      out = out + "\n";
      out = out + "Score: ";
      out = out + std::to_string(result.score);
      out = out + "\n";
    }
    //std::cout<<"C++ output:"<<std::endl<<out<<std::endl;
    return out;
    
    
  }

}

