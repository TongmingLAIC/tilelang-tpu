// clang-format off
#pragma once
#if !defined(__sg2260__) && !defined(__bm1690e__)
#error "TPU chip macro is required; compile generated wrappers through PPLLayout/LibraryGenerator"
#endif

#include <stdint.h>
#include <assert.h>
#ifdef __cplusplus
extern "C" {{
#endif

typedef struct {{
  {struct_members}
}} tpu_kernel_api_{function_name}_t;

tpu_kernel_api_{function_name}_t fill_{function_name}_struct({struct_params});

int {function_name}_check_mem({func_params});

int {function_name}_check_mem_s(tpu_kernel_api_{function_name}_t *api);

int {function_name}({func_params});

#ifdef __cplusplus
}}
#endif
// clang-format on
