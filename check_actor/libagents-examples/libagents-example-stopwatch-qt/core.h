#ifndef CORE_H
#define CORE_H

#include "libagents.h"
#include "globals.h"

class StopwatchCore: public libagents::Core {
   class StopwatchTask *stopwatchTask;

public:
   class StopwatchShellController *sc;
   StopwatchCore();
   int onStarted(const libagents::message_t &args);
   bool onIoMessage(const libagents::message_t &msg);
};

#endif // CORE_H
