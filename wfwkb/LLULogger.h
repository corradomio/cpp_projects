//
// Created by Corrado Mio on 21/07/2023.
//

#ifndef WFWKB_LLULOGGER_H
#define WFWKB_LLULOGGER_H

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#define LLU_LEVEL_DEBUG     spdlog::level::debug

#ifdef LLU_LOG_DEBUG

#define LLU_LOG_INITIALIZE(name,file) \
    auto logger = spdlog::basic_logger_mt(name, file);

#define LLU_LOG_LEVEL(l)    spdlog::set_level(spdlog::level::debug);

#define LLU_DEBUG(...)      logger->debug(__VA_ARGS__)
#define LLU_INFO(...)       logger->info(__VA_ARGS__)
#define LLU_WARNING(...)    logger->warn(__VA_ARGS__)
#define LLU_ERROR(...)      logger->error(__VA_ARGS__)

#else

#define LLU_LOG_INITIALIZE(name,file)
#define LLU_LOG_LEVEL(l)
#define LLU_DEBUG(msg,...)
#define LLU_WARNING(msg,...)
#define LLU_ERROR(msg,...)

#endif


#endif //WFWKB_LLULOGGER_H
