
/*
    File to be linked to C code, in order to use C++ functions
*/

#ifdef __cplusplus
extern "C" {
#endif

char* classify_image(
                char* model_path,
                char* image_path,
                char* labels_path,
                float input_mean,
                float input_std
            );

#ifdef __cplusplus
}
#endif