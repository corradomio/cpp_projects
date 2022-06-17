#include "main_window.h"
#include "ui_main_window.h"

#include "shell_controller.h"

using namespace libagents;

MainWindow::MainWindow(QWidget *parent) :
   QMainWindow(parent),
   ui(new Ui::MainWindow)
{
   ui->setupUi(this);
   setFixedSize(size());
   window=ui;
}

MainWindow::~MainWindow()
{
   delete ui;
}

void MainWindow::DisplayMinutes(int minutes)
{
   window->minutesDisplay->display(minutes);
}

void MainWindow::DisplaySeconds(int seconds)
{
   window->secondsDisplay->display(seconds);
}

void MainWindow::on_startStopButton_clicked()
{
   assert(sc->SendIoMessage((message_t)STARTSTOP_BUTTON_PRESSED));
}

void MainWindow::on_resetButton_clicked()
{
   assert(sc->SendIoMessage((message_t)RESET_BUTTON_PRESSED));
}
