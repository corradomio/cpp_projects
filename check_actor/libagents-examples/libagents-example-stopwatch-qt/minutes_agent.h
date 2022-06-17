#ifndef MINUTESAGENT_H
#define MINUTESAGENT_H

#include "libagents.h"
#include "globals.h"

class MinutesAgent: public libagents::Agent {
   int minutes;
public:
   MinutesAgent(libagents::compid_t id);
   void onStarted();
   bool onMessage(const libagents::message_t &msg, const libagents::compid_t &sourceThreadId, const libagents::compid_t &sourceAgentId);
};

#endif // MINUTESAGENT_H
