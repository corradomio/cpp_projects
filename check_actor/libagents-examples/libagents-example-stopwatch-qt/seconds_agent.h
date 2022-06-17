#ifndef SECONDSAGENT_H
#define SECONDSAGENT_H

#include "libagents.h"
#include "globals.h"

class SecondsAgent: public libagents::Agent {
   int seconds;
public:
   SecondsAgent(libagents::compid_t id);
   void onStarted();
   bool onMessage(const libagents::message_t &msg, const libagents::compid_t &sourceThreadId, const libagents::compid_t &sourceAgentId);
};

#endif // SECONDSAGENT_H
