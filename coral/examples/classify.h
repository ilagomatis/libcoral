
/*
    File to be linked to C code, in order to use C++ functions
*/
#ifndef __CLASSIFY_C_H__
#define __CLASSIFY_C_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

char* classify_image(
                char* model_path,
                char* image_path,
                char* labels_path,
                float input_mean,
                float input_std
            );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __CLASSIFY_C_H__ */
