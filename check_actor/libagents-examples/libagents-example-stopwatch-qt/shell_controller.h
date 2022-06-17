#ifndef SHELL_CONTROLLER_H
#define SHELL_CONTROLLER_H

#include "libagents.h"
#include "globals.h"

#include <QWidget>
#include <QTimer>

namespace Ui {
class StopwatchShellController;
}

// the ShellController is implemented as a Qt Form in order to be able to instantiate and connect Qt's QTimer
class StopwatchShellController : public QWidget
{
   Q_OBJECT
private:
   Ui::StopwatchShellController *ui;
   QTimer ticker;
private slots:
   void tickerTick(); // event handler for the QTimer's 'timeout' signal
public:
   explicit StopwatchShellController(QWidget *parent = 0);
   ~StopwatchShellController();
   class StopwatchCore *core;
   class MainWindow *mw;
   void ShellExec();
   bool SendIoMessage(const libagents::message_t &msg);
   void Start();
};

#endif // SHELL_CONTROLLER_H
