cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
# 小智语音模块的结构体初始化写法依赖这个开关（和小智工程一致）
add_compile_options(-Wno-missing-field-initializers)
project(ai_pet_fw)
