#ifndef CLOCKTHREAD_H
#define CLOCKTHREAD_H

#include "libagents.h"
#include "globals.h"

class ClockThread: public libagents::Thread {
   class ClockAgent *clockAgent;
public:
   ClockThread(libagents::compid_t id);
   void onStarted();

};

#endif // CLOCKTHREAD_H
