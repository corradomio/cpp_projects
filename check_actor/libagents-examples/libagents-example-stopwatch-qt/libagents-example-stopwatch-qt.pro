#-------------------------------------------------
#
# Project created by QtCreator 2014-11-09T13:52:11
#
#-------------------------------------------------

CONFIG += c++11 # manually inserted in order to enable C++11

INCLUDEPATH += "../../"

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = libagents-example-stopwatch-qt
TEMPLATE = app


SOURCES +=\
    main.cpp \
    core.cpp \
    clock_agent.cpp \
    clock_thread.cpp \
    counters_thread.cpp \
    minutes_agent.cpp \
    seconds_agent.cpp \
    stopwatch_task.cpp \
    main_window.cpp \
    shell_controller.cpp \
    ../../libagents.cpp

HEADERS  += \
    globals.h \
    core.h \
    clock_agent.h \
    clock_thread.h \
    counters_thread.h \
    minutes_agent.h \
    seconds_agent.h \
    stopwatch_task.h \
    main_window.h \
    shell_controller.h \
    ../../libagents.h \
    ../../libagents-config.h

FORMS    += \
    main_window.ui \
    shell_controller.ui

OTHER_FILES +=

DISTFILES += \
    README.TXT
