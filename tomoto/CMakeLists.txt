cmake_minimum_required(VERSION 3.23)
project(tomoto)

include_directories(include src)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj -march=tigerlake -O1")

set(SOURCES
    main.cpp
    src/Labeling/FoRelevance.cpp
    src/TopicModel/CTModel.cpp
    src/TopicModel/DMRModel.cpp
    src/TopicModel/DTModel.cpp
    src/TopicModel/GDMRModel.cpp
    src/TopicModel/HDPModel.cpp
    src/TopicModel/HLDAModel.cpp
    src/TopicModel/HPAModel.cpp
    src/TopicModel/LDAModel.cpp
    src/TopicModel/LLDAModel.cpp
    src/TopicModel/MGLDAModel.cpp
    src/TopicModel/PAModel.cpp
    src/TopicModel/PLDAModel.cpp
    src/TopicModel/PTModel.cpp
    src/TopicModel/SLDAModel.cpp
)

add_executable(tomoto ${SOURCES})
