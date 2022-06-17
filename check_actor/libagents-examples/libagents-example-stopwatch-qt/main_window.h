#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "globals.h"
#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
   Q_OBJECT

private:
   Ui::MainWindow *ui;

private slots:
   void on_startStopButton_clicked(); // Qt Start/Stop push-button's event handler
   void on_resetButton_clicked();     // Qt Reset push-button's event handler

public:
   class StopwatchShellController *sc;
   explicit MainWindow(QWidget *parent = 0);
   ~MainWindow();
   Ui::MainWindow *window;
   void DisplayMinutes(int minutes);
   void DisplaySeconds(int seconds);
};

#endif // MAINWINDOW_H
