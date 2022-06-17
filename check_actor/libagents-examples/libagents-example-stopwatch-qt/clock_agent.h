#ifndef CLOCKAGENT_H
#define CLOCKAGENT_H

#include "libagents.h"
#include "globals.h"

class ClockAgent: public libagents::Agent {
   int running;
   int tickRate;
   uint64_t autorepeatMsgCode;
public:
   ClockAgent(libagents::compid_t id, int tickRate);
   void onStarted();
   bool onMessage(const libagents::message_t &msg, const libagents::compid_t &sourceThreadId, const libagents::compid_t &sourceAgentId);
};

#endif // CLOCKAGENT_H
