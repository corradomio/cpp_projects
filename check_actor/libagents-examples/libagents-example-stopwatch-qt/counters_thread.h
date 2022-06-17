#ifndef COUNTERSTHREAD_H
#define COUNTERSTHREAD_H

#include "libagents.h"
#include "globals.h"

class CountersThread: public libagents::Thread {
   class MinutesAgent *minutesAgent;
   class SecondsAgent *secondsAgent;
public:
   CountersThread(libagents::compid_t id);
   void onStarted();
};

#endif // COUNTERSTHREAD_H
