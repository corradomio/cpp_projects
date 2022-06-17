#include <QApplication>

#include "globals.h"
#include "core.h"
#include "shell_controller.h"
#include "main_window.h"

using namespace libagents;

int main(int argc, char *argv[])
{
   // host framework INIT (mandated by the Qt framework)
   QApplication a(argc, argv);

   // instatiate the Core module object
   StopwatchCore core;
   // instatiate the Shell module objects
   StopwatchShellController sc;
   MainWindow mw;

   // cross-link the ShellController (i.e. 'sc') with the ShellObjects (i.e. 'mw') and the app's Core object (i.e. 'core')
   sc.core=&core;
   core.sc=&sc;
   sc.mw=&mw;
   mw.sc=&sc;

   // sart the app's Core and ShellCotroller (in this order)
   core.Start();
   sc.Start();

   // host framework RUN (mandated by the Qt framework)
   a.exec();

   // finished
   return 0;
}
