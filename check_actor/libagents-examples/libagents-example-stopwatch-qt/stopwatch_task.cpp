#include "stopwatch_task.h"

#include "clock_thread.h"
#include "counters_thread.h"

using namespace libagents;

StopwatchTask::StopwatchTask(compid_t id): Task(id)
{
   countersThread = new CountersThread(COUNTERS_THREAD);
   clockThread = new ClockThread(CLOCK_THREAD);
}

void StopwatchTask::onStarted()
{
   // start the coutersThread before the clockThread to ensure that the clockAgent won't send messages to a not-yet-started secondsAgent
   StartThread(countersThread);
   StartThread(clockThread);
}
