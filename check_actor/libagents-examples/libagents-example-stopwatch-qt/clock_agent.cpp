#include "clock_agent.h"

using namespace libagents;

ClockAgent::ClockAgent(compid_t id, int tickRate): Agent(id)
{
   running=0;
   this->tickRate=tickRate;
}

void ClockAgent::onStarted()
{
}

bool ClockAgent::onMessage(const message_t &msg, const compid_t &sourceThreadId, const compid_t &sourceAgentId)
{
   assert(sourceThreadId=="" && sourceAgentId==""); // the messages to the clockAgent can come only from the Core object's OutputMessage()
   switch(msg[0].integer()) { // all the message names in this app are declared as enums
   case STARTSTOP_CLOCK:
      running = !running;
      break;
   default:
      assert(0);
   }
   if (running) {
      assert(autorepeatMsgCode=SendMessage(SECONDS_TICK, COUNTERS_THREAD, SECONDS_AGENT, tickRate, INT_MAX)); // send an auto-repeating message every tickRate milliseconds to the secondsAgent counter
   }
   else {
      CancelMessage(autorepeatMsgCode);
   }
   return 1; // libagents v2.0.x requires this method to return 'true'
}
