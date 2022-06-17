#include "core.h"

#include "stopwatch_task.h"
#include "shell_controller.h"

using namespace libagents;

StopwatchCore::StopwatchCore(): Core()
{
   stopwatchTask = new StopwatchTask(STOPWATCH_TASK);
}

int StopwatchCore::onStarted(const message_t &args)
{
   StartTask(stopwatchTask);
   return 1; // this value is passed to, and then returned by, the StopwatchCore::Start() method
}

bool StopwatchCore::onIoMessage(const message_t &msg)
{
   switch(msg[0].integer()) { // all the message names in this app are declared as enums
   case STARTSTOP_BUTTON_PRESSED:
      assert(SendMessage(STARTSTOP_CLOCK, STOPWATCH_TASK, CLOCK_THREAD, CLOCK_AGENT));
      break;
   case RESET_BUTTON_PRESSED:
      assert(SendMessage(RESET_COUNTER, STOPWATCH_TASK, COUNTERS_THREAD, SECONDS_AGENT)); // start with secondsAgent
      assert(SendMessage(RESET_COUNTER, STOPWATCH_TASK, COUNTERS_THREAD, MINUTES_AGENT));
      break;
   default:
      assert(0);
   }
   return 1; // libagents v2.0.x requires this method to return 'true'
}
