#include "counters_thread.h"

#include "minutes_agent.h"
#include "seconds_agent.h"

using namespace libagents;

CountersThread::CountersThread(compid_t id): Thread(id)
{
   minutesAgent = new MinutesAgent(MINUTES_AGENT);
   secondsAgent = new SecondsAgent(SECONDS_AGENT);
}

void CountersThread::onStarted()
{
   // start minutesAgent before secondsAgent such that the secondsAgent doesn't send message to not-yet-started minutesAgent
   StartAgent(minutesAgent);
   StartAgent(secondsAgent);
}
