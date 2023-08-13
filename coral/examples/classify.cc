
#include "coral/examples/classify_image.h"
#include "coral/examples/classify.h"
#include<cstdio>
#include<cstdlib>
#include<cstring>
extern "C"
char* classify_image(char* model_path,
                     char* image_path,
                     char* labels_path,
                     float input_mean,
                     float input_std
                    )
    {
        std::string out = classify::classifyImage
                    (
                        std::string(model_path),
                        std::string(image_path),
                        std::string(labels_path),
                        input_mean,
                        input_std
                    );

        char* output = (char*)malloc((out.std::string::length() + 1) * sizeof(char));
        std::strcpy(output, out.std::string::c_str());
        //printf("\nC out: \n%s\n", output);
        return output;
    }