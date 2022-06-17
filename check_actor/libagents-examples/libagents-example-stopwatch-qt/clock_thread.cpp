#include "clock_thread.h"

#include "clock_agent.h"

using namespace libagents;

ClockThread::ClockThread(compid_t id): Thread(id)
{
   clockAgent = new ClockAgent(CLOCK_AGENT, CLOCK_TICK);
}

void ClockThread::onStarted()
{
   StartAgent(clockAgent);
}
