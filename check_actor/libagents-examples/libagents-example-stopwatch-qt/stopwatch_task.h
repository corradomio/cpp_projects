#ifndef STOPWATCHTASK_H
#define STOPWATCHTASK_H

#include "libagents.h"
#include "globals.h"

class StopwatchTask: public libagents::Task {
   class ClockThread *clockThread;
   class CountersThread *countersThread;
public:
   StopwatchTask(libagents::compid_t id);
   void onStarted();
};

#endif // STOPWATCHTASK_H
