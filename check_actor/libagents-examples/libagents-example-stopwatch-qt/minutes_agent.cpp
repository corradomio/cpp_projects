#include "minutes_agent.h"

using namespace libagents;

MinutesAgent::MinutesAgent(compid_t id): Agent(id)
{
   minutes=0;
}

void MinutesAgent::onStarted()
{
   assert(core->SendIoMessage((message_t)DISPLAY_MINUTES << minutes));
}

bool MinutesAgent::onMessage(const message_t &msg, const compid_t &sourceThreadId, const compid_t &sourceAgentId)
{
   switch(msg[0].integer()) { // all the message names in this app are declared as enums
   case MINUTES_TICK:
      minutes++;
      break;
   case RESET_COUNTER:
      minutes=0;
      break;
   default:
      assert(0);
   }
   assert(core->SendIoMessage((message_t)DISPLAY_MINUTES << minutes));
   return 1; // libagents v2.0.x requires this method to return 'true'
}
