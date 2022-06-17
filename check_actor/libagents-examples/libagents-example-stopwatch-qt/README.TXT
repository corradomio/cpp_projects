The 'libagents-example-stopwatch-qt' application
================================================

Project setup:
--------------
1) the compiler must be set up to support C++11
2) the compiler's #include path must contain the *full path* of the 'libagents-x.y.z' base folder (i.e. the path must be of the form: '<-path-to->/libagents-x.y.z')
3) a Qt-5.9.5 project file 'stopwatch.pro' is readily-available in the 'libagents-2.0.x/libagents-examples/libagents-example-stopwatch-qt' folder


Notations used in this doc:
---------------------------
1) objectName: class ObjectType: BaseClass (the BaseClass of ObjectType is shown only if ObjectType is a libagents type)
2) containerName
   |
   +---> objectName (objectName is a component of containerName)


Core module components:
-----------------------
core: class StopwatchCore: libagents::Core (this is the stopwatch application's Core object)
|
+--->stopwatchTask: class StopwatchTask: libagents::Task (this is the only Task object in the application)
     |
     +---> clockThread: class ClockThread: libagents::Thread
     |     |
     |     +---> clockAgent: class ClockAgent: libagents::Agent (this Agent generates a tick every one second)
     |
     +---> countersThread: class CountersThread: libagents::Thread
           |
           +---> secondsAgent: class SecondsAgent: libagents::Agent (this Agent counts and displays the seconds)
           +---> minutesAgent: class MinutesAgent: libagents::Agent (this Agent counts and displays the minutes)

Notes:
- the dual-thread architecture above has been chosen to illustrate how multiple Agent objects can be organized as components of different threads; however, other arrangements could also have been made, e.g. the ClockAgent, the SecondsAgent and the MinutesAgent could have been all placed in a single thread, or they could have been placed each in a separate thread


Shell module components:
------------------------
main(): the application's startup function (instantiates all the Shell module components and the application's Core object)

sc: class StopwatchShellController (the shell controller object, implemented as an invisible QWidget)
|
+---> ticker: class QTimer (object that implements the polling ticker object; polling interval set by MESSAGE_POLLING_TIMER_TICK)

mw: class MainWindow: QMainWindow (the application shell's main window containing the control buttons and the clock display)
|
+---> startStopButton: class QPushButton
+---> resetButton: class QPushButton
+---> minutesDisplay: class QLCDNumber
+---> secondsDisplay: class QLCDNumber


Event Handlers and Algorithms:
------------------------------
MainWindow object:
- MainWindow::on_startStopButton_clicked: event handler for pressing the Start/Stop GUI button
  - invokes StopwatchShellController::SendIoMessage(STARTSTOP_BUTTON_PRESSED)
- MainWindow::on_resetButton_clicked: event handler for pressing the Reset GUI button
  -  invokes StopwatchShellController::SendIoMessage(RESET_BUTTON_PRESSED)
- MainWindow::DisplaySeconds: displays the received value in the secondsDisplay on the main window
- MainWindow::DisplayMinutes: displays the received value in the minutesDisplay on the main window

Core object:
- StopwatchCore::onIoMessage: triggered for each message that is received by the Core object
  - parses the message received from the Shell module, and then, depending on the message, performs a specific action; specifically:
    - for STARTSTOP_BUTTON_PRESSED:
      - sends a STARTSTOP_CLOCK message to the clockAgent
    - for RESET_BUTTON_PRESSED:
      - sends a RESET_COUNTER message to the secondsAgent
      - sends a RESET_COUNTER message to the minutesAgent

ClockAgent:
- ClockAgent::onMessage: triggered when the clockAgent receives a message
  - parses the messages received by the clockAgent, and then, depending on the message, performs a specific action; specifically:
    - for STARTSTOP_CLOCK:
      - the clockAgent's 'running' state is toggled between true/false
- while running==true the clockAgent sends a SECONDS_TICK message to the secondsAgent every 1 second

SecondsAgent:
- SecondsAgent::onMessage: triggered when the secondsAgent receives a message
  - parses the messages received by the secondsAgent, and then, depending on the message, performs a specific action; specifically:
    - for SECONDS_TICK:
      - increments the seconds counter
      - if seconds==60 then:
        - resets the seconds couter
        - sends a MINUTES_TICK message to the minutesAgent
      - invokes StopwatchCore::ReadIoMessage(DISPLAY_SECONDS): sends a DISPLAY_SECONDS message (with value 0) to the Shell module
    - for RESET_COUNTER:
      - resets the seconds counter
      - invokes StopwatchCore::ReadIoMessage(DISPLAY_SECONDS): sends a DISPLAY_SECONDS message (with value 0) to the Shell module

MinutesAgent:
- MinutesAgent::onMessage: triggered when the minutesAgent receives a message
  - parses the messages received by the minutesAgent, and then, depending on the message, performs a specific action; specifically:
    - for MINUTES_TICK:
      - increments the minutes counter
      - invokes StopwatchCore::ReadIoMessage(DISPLAY_MINUTES): sends a DISPLAY_MINUTES message (with the value 0) to the Shell module
    - for RESET_COUNTER:
      - resets the minutes counter
      - invokes StopwatchCore::ReadIoMessage(DISPLAY_MINUTES): sends a DISPLAY_MINUTES message (with the value 0) to the Shell module

StopwatchShellController:
- StopwatchShellController::ShellExec(): triggered by the ticker object, at a rate of MESSAGE_POLLING_TIMER_TICK/sec
  - invokes the Corce::DeliverIoMessage() method to extract the messages queued in the Core object Intercom's Output Buffer (if any)
    - MESSAGE_POLLING_BLOCK_SIZE determines the maximum number of messages that are extracted and processed at each tick before ShellExec() returns (ie. before giving back control to the GUI event loop)
      - MESSAGE_POLLING_BLOCK_SIZE must be small enough to ensure that the GUI thread does not freeze for too long when a new set of messages is extracted and processed
      - MESSAGE_POLLING_BLOCK_SIZE must be large enough to ensure that the Core object's intercom buffer doesn't get full for the highest rate of messages sent by the Core object
      - for this application MESSAGE_POLLING_BLOCK_SIZE can be 1 because the processing requirements/message are very low, and the messages have a very low rate (one/sec)
  - parses the extracted message and then, depending on the message, performs a specific action; specifically:
    - for DISPLAY_MINUTES:
      - invokes the MainWindow::DisplayMinutes() method: displays the minutes in the minutesDisplay LCD panel
    - for DISPLAY_SECONDS:
      - invokes the MainWindow::DisplaySeconds() method: displays the seconds in the secondsDisplay LCD panel
- StopwatchShellController::SendIoMessage(): called by the MainWindow buttons' event handlers
  - invokeds the Corce::ReadIoMessage() method to place the message (specified in its argument) in the Core object Intercom's Input Buffer
