#include "shell_controller.h"
#include "ui_shell_controller.h"

#include "core.h"
#include "main_window.h"

using namespace libagents;

StopwatchShellController::StopwatchShellController(QWidget *parent) :
   QWidget(parent),
   ui(new Ui::StopwatchShellController)
{
   ui->setupUi(this);
}

StopwatchShellController::~StopwatchShellController()
{
   delete ui;
}

void StopwatchShellController::Start()
{
   mw->show();
   ticker.setInterval(MESSAGE_POLLING_TIMER_TICK);
   connect(&ticker, SIGNAL(timeout()) , this, SLOT(tickerTick()));
   ticker.start(); }
   void StopwatchShellController::tickerTick() {
	ShellExec();
}

void StopwatchShellController::ShellExec()
{
   message_t msg;
   for (int i=0; i<MESSAGE_POLLING_BLOCK_SIZE && core->DeliverIoMessage(&msg); i--) {
      switch(msg[0].integer()) { // all the message names in this app are declared as enums
      case DISPLAY_SECONDS:
         mw->DisplaySeconds(msg[1].integer());
         break;
      case DISPLAY_MINUTES:
         mw->DisplayMinutes(msg[1].integer());
         break;
      default:
         assert(0);
      }
   }
}

bool StopwatchShellController::SendIoMessage(const message_t &msg)
{
   return core->ReadIoMessage(msg);
}
