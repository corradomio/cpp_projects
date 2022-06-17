#include "seconds_agent.h"

using namespace libagents;

SecondsAgent::SecondsAgent(compid_t id): Agent(id)
{
   seconds=0;
}

void SecondsAgent::onStarted()
{
   assert(core->SendIoMessage((message_t)DISPLAY_SECONDS << seconds));
}

bool SecondsAgent::onMessage(const message_t &msg, const compid_t &sourceThreadId, const compid_t &sourceAgentId)
{
   switch(msg[0].integer()) { // all the message names in this app are declared as enums
   case SECONDS_TICK:
      seconds++;
      if (seconds==60) { // every 60 seconds reset the seconds counter and send a message to increment the minutesAgent
         seconds=0;
         assert(SendMessage(MINUTES_TICK, COUNTERS_THREAD, MINUTES_AGENT));
      }
      break;
   case RESET_COUNTER:
      seconds=0;
      break;
   default:
      assert(0);
   }
   assert(core->SendIoMessage((message_t)DISPLAY_SECONDS << seconds));
   return 1; // libagents v2.0.x requires this method to return 'true'
}
