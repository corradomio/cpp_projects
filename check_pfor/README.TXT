//////////////////////////////////////////////////////////
// THIS FILE IS PART OF THE 'libagents' LIBRARY PACKAGE //
//////////////////////////////////////////////////////////

/*
 * The 'libagents' library is copyrighted software, all rights reserved.
 * (c) Information Technology Group - www.itgroup.ro
 * (c) Virgil Mager
 *
 * THE 'libagents' LIBRARY IS LICENSED FREE OF CHARGE, AND IT IS DISTRIBUTED "AS IS",
 * WITH EXPRESS DISCLAIMER OF SUITABILITY FOR ANY PARTICULAR PURPOSE,
 * AND WITH NO WARRANTIES OF ANY KIND, WHETHER EXPRESS OR IMPLIED,
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW.
 *
 * IN NO EVENT OTHER THAN WHEN MANDATED BY APPLICABLE LAW, SHALL THE AUTHOR(S),
 * THE PUBLISHER(S), AND/OR THE COPYRIGHT HOLDER(S) OF THE 'libagents' LIBRARY,
 * BE LIABLE FOR ANY CONSEQUENTIAL DAMAGES WHICH MAY DIRECTLY OR INDIRECTLY RESULT
 * FROM THE USE OR MISUSE, IN ANY WAY AND IN ANY FORM, OF THE 'libagents' LIBRARY.
 *
 * Permission is hereby granted by the 'libagents' library copyright holder(s)
 * to create and/or distribute any form of work based on, and/or dervied from,
 * and/or which uses in any way, the 'libagents' library, in full or in part, provided that
 * ANY SUCH DISTRUBUTION PROMINENTLY INCLUDES THE ORIGINAL 'libagents' LIBRARY PACKAGE.
 */

The libagents library provides a C++ implementation of the "Actor Model" paradigm
(see http://en.wikipedia.org/wiki/Actor_model), thus enabling the development
of pure C++ multi-threaded applications structured as a collection of
asynchronous event-driven execution agents which all run concurrently
and which communicate with each other via an asynchronous messaging system.


Versioning
==========
The libagents releases follow the standard versioning format: major.minor.revision
  - major: incremental major numbers are not backward compatible
  - minor: incremental minor numbers provide new features, without breaking bakward compatibility among releases with the same major number
  - revision: incremental revision numbers provide bug fixes, code refactoring, and/or improvements in the documentation, and they are fully compatible within the same major.minor releases
Occasionally, a new "silent release" will replace a previous release with the same version number (i.e. the same major.minor.revision) if the changes that have been introduces do not change any functionality and are considered not significant enough for a new version number


Build Settings
=============
- any project that uses any the libagents library must:
  - add the 'libagents.h', 'libagents.cpp', and 'libagents-config.h', found in the 'libagents-x.y.z' base folder, into the project sructure
  - set the compiler to support C++11 (or later)
  - set the compiler's #include path to contain the 'libagents-x.y.z' base folder (e.g. "C:\libagents-x.y.z\" if the libagents folder is placed on the root level of a Windows system's C: drive)

Qt Creator
----------
if using Qt Creator, the following special configuation settings are necessary in the project's .pro file for using the libagents library:
CONFIG += C++11
INCLUDEPATH += "<full>/<path>/<to>/libagents-x.y.z/" # path must end with the libagents folder (i.e. with "/libagents-x.y.z/")


Folder Structure
================
{libagents-x.y.z}
|
+---libagents.h
+---libagents.cpp
+---libagents-config.h


Documentation
=============
The libagents reference manual is issued in OpenDocument HybridPDF format which is readable in any modern PDF reader and it is editable with LibreOffice 4+ office suite
