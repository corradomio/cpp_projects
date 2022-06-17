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

#ifndef __AGENTS_Lib__
#define __AGENTS_Lib__

#include "libagents-config.h"
#include <map>
#include <string>
#include <deque>
#include <mutex>
#include <unordered_set>
#include <tuple>

namespace AGENTS_Lib {

class data_t;
class message_t;
class Task;
class Thread;
class Agent;
class Core;

class data_t {
   friend class message_t; // uses t's enum
   enum {undefined,ts,td,tp} t=undefined;
   std::string cs, cs_tmp;
   double cd=0;
   void *cp=nullptr;
   constexpr static const char *reserved="#$^|* (?!%)"; // the characters (?!%) are reserved but not used in this version
public:
   inline data_t() = default;
   inline data_t(const char *s) {t=ts; cp=nullptr; cd=0; cs=s;}
   inline data_t(const std::string &s) {t=ts; cp=nullptr; cd=0; cs=s;}
   inline data_t(double d) {t=td; cp=nullptr; cd=d; cs[0]=0;}
   inline data_t(int i) {t=td; cp=nullptr; cd=(double)i; force((int)cd==i); cs[0]=0;} // 'force()' checks that the int value does not lose precision
   inline data_t(unsigned u) {t=td; cp=nullptr; cd=(double)u; force((unsigned)cd==u); cs[0]=0;} // 'force()' checks that the unsigned value does not lose precision
   inline data_t(void *p) {t=tp; cp=p; cd=0; cs[0]=0;}
   inline bool isValidId() const {
      if (t==ts) return cs.size() && cs.find_first_of(reserved)==std::string::npos;
      else if (t==td) return 1;
      else if (t==tp) return 0;
      else {assert(0); return 0;}}
   inline bool operator==(const data_t &x) const {
      if (t!=x.t) return 0;
      else if (t==data_t::ts) return cs==x.cs;
      else if (t==data_t::td) return cd==x.cd;
      else if (t==data_t::tp) return cp==x.cp;
      else {assert(0); return 0;}}
   inline bool operator!=(const data_t &x) const {return !(*this==x);}
   inline bool is_number() const {return t==td;}
	inline bool is_integer() {return t==td && (int)cd==cd;}
   inline bool is_string() const {return t==ts;}
   inline bool is_pointer() const {return t==tp;}
   inline int integer() const {int i=(int)cd; force(t==td && i == cd); return i;}
   inline double number() const {force(t==td); return cd;}
   inline void *pointer() const {force(t==tp); return cp;}
   inline const std::string &string() const {force(t==ts); return cs;}
   inline const std::string &to_unique_str() const { // ### the unique string is a host system-dependent INTERNAL REPRESENTATION, it is NOT UNIQUE ACROSS DIFFERENT HOST SYSTEMS and it should NEVER BE TRANSMITTED ACROSS DIFFERENT SYSTEMS
      if (t==ts) {
         const_cast<std::string&>(cs_tmp)=cs+"$";
         return cs_tmp;
      }
      else if (t==td) {
         const_cast<std::string&>(cs_tmp)=(cd==(int)cd? std::to_string((int)cd): std::to_string(cd))+"#";
         return cs_tmp;
      }
      else if (t==tp) {
         const_cast<std::string&>(cs_tmp)=std::to_string((unsigned long long)cp)+"^";
         return cs_tmp;
      }
      else {
         assert(0); return cs_tmp;
      }
   }
};

typedef data_t compid_t; // use ONLY 'compid_t' for core component id's (tasks, threads, agents)

class message_t {
   friend class Task;
   friend class Thread;
   friend class Agent;
   friend class Core;
   friend class MessageBuffer;
   std::deque<data_t> qmsg;
   compid_t sourceThreadId, sourceAgentId, destThreadId, destAgentId;
   compid_t sourceTaskId; // only used when the message is sent to the parent Core's txBuffer
   enum {undefined, scheduled, replicated, prioritized} type=undefined;
   int repeatCount=-1;
   float deliverySchedule=-1;
   int expireThreshold=-1;
   Agent *sourceAgent=nullptr;
   Thread *sourceThread=nullptr;
public:
   std::map<int,data_t> meta;
   message_t() = default;
   message_t(const std::string &s);
   message_t(const char *c);
   message_t(double d);
   message_t(int i);
   message_t(unsigned u);
   message_t(void *p);
   message_t(const data_t &a);
   void clear();
// const message_t &operator=(const strnum_t &smsg);
   const data_t &operator[](unsigned i) const;
   data_t &operator[](unsigned i);
   message_t &operator<<(const data_t &p);
   message_t &insert(unsigned pos, const data_t &p);
   bool erase(unsigned pos);
   unsigned size() const;
   std::string prettyPrint(std::string separator="", bool decoration=0) const; // resulting string may vary depnding on the host system
   bool to_string(std::string &smsg) const; // ### the encode->decode sequence from_string(to_string()) may introduce rounding errors for 'double' data cells, more so when encode->decode are done on different-precision host systems
   bool from_string(const std::string &smsg, int cap=0);
};

struct msgix_t {
   uint64_t l,h;
   inline msgix_t() {h=l=0;}
   inline msgix_t(const msgix_t &x) {h=x.h; l=x.l;}
   inline msgix_t(uint64_t x) {h=0; l=x;}
   inline bool operator<(const msgix_t &x) const {if (h==x.h) return l<x.l; else return h<x.h;}
   inline operator bool() {return h||l;}
   inline std::string to_string() {return std::to_string(h)+"_"+std::to_string(l)+"#";} // std::to_string(l/h) converts l/h from uint64_t to unsigned long long which is OK because unsigned long long is guaranteed >=64 bits
};

struct MessageBuffer {
   std::recursive_mutex editMessageBuffer;
   std::map<msgix_t,message_t> messages; // data stays always sorted by time because it's std::map
	unsigned messageBufferSize=0;
   uint64_t PushMessage(const message_t &msg, msgix_t msgIx=0); // returns messageId, or 0 if error
   uint64_t PopMessage(message_t &msg); // returns messageId, or 0 if error
};

// !!! CRITICAL: Agents CANNOT use blocking functions (e.g. sleep(), future::get(), etc) to wait for a condition; instead, they have to schedule a retry message to themselves for a later time
class Agent {
   friend class Thread;
   friend class Task;
   friend class Core;
   friend class MessageBuffer;
   compid_t id;
   static uint32_t agentCounter;
   Thread *parent=nullptr;
   bool started=0;
   std::mutex editMessageTracker;
   std::map<uint64_t,std::pair<MessageBuffer*,msgix_t>> messageTracker;
   class {
      bool waiting=0;
      std::string messageName;
      std::string sourceAgentId;
      std::string sourceThreadId;
   } wait;
protected:
   virtual void onStarted() = 0;
   virtual bool onMessage(const message_t &msg, const compid_t &sourceThreadId, const compid_t &sourceAgentId)=0;
   uint64_t SendMessage(const message_t &msg, const compid_t &destThreadId, const compid_t &destAgentId, float schedule=0, int repeat=1, int expire=INT_MAX);
   uint64_t BroadcastMessage(const message_t &msg, const compid_t &destThreadId="*", float schedule=0, int repeat=1, int expire=INT_MAX);
   bool CancelMessage(uint64_t msgId);
public:
   Agent(const compid_t &id);
   virtual ~Agent();
   compid_t agentId();
   Thread *parentThread();
   Core *core;
};

class Thread {
   friend class Task;
   friend class Agent;
   friend class Core;
   friend class MessageBuffer;
   compid_t id;
   Task *parent=nullptr;
   std::map<std::string,Agent*> registeredAgents;
   int DispatchMessage(message_t &msg);
   bool runSwitch=0, isRunning=0;
   void Start();
   void Stop();
   void messageLoop();
   Agent *operator[](const compid_t &agentId);
   std::recursive_mutex editAgents;
   MessageBuffer messageBuffer;
   bool enableCancelMessage=1; // i only used it for various debugging; i used it to see what performance impact it has to store messages in the message-canceling data structure
protected:
   virtual void onStarted() = 0;
public:
   Thread(const compid_t &id, unsigned messageBufferSize=MESSAGE_BUFFER_SIZE);
   virtual ~Thread();
   bool StartAgent(Agent *a);
   bool KillAgent(const compid_t &agentId);
   Agent *childAgent(const compid_t &agentId);
   compid_t threadId();
   Task *parentTask();
   Core *core;
};

class Task {
   friend class Thread;
   friend class Agent;
   friend class Core;
   compid_t id;
   Core *parent;
   std::map<std::string,Thread*> registeredThreads;
   Thread *operator[](const compid_t &threadId);
   bool RelayMessage(const message_t &msg);
   std::recursive_mutex editThreads;
   std::mutex editMessageRoutingTable;
   std::multimap<std::string, std::tuple<compid_t,compid_t>> messageRoutingTable;
   static std::string messageRoutingTableIndex(const data_t &messageName, const compid_t &sourceThreadId, const compid_t &sourceAgentId);
   int RemoveAgentRouting(const compid_t &threadId, const compid_t &agentId);
protected:
   virtual void onStarted() = 0;
public:
   Task(const compid_t &id);
   virtual ~Task();
   bool StartThread(Thread *t);
   bool KillThread(const compid_t &threadId);
   Thread *childThread(const compid_t &threadId);
   compid_t taskId();
	Core* parentCore();
   Core *core;
   bool AddBroadcastSubscription(const data_t &messageName, const compid_t &sourceThreadId, const compid_t &sourceAgentId, const compid_t &destThreadId, const compid_t &destAgentId);
   bool RemoveBroadcastSubscription(const data_t &messageName, const compid_t &sourceThreadId, const compid_t &sourceAgentId, const compid_t &destThreadId, const compid_t &destAgentId);
};

class Core {
   friend class Agent;
   std::map<std::string,Task*> registeredTasks;
   Task *operator[](const compid_t &taskId);
   std::recursive_mutex editTasks;
	class Intercom {
      friend class Core;
      Intercom(Core *core);
      ~Intercom();
      Core *core;
      MessageBuffer rxBuffer;
      MessageBuffer txBuffer;
      bool inputMonitorRunSwitch=0;
      bool inputMonitorIsRunning=0;
      void inputMonitor();
      bool PushMessage(const message_t &msg);
      bool PopMessage(message_t &msg);
   } intercom;
protected:
   virtual int onStarted(const message_t &args="") = 0;
   bool SendMessage(const message_t &msg, const compid_t &destTaskId, const compid_t &destThreadId, const compid_t &destAgentId);
   virtual bool onIoMessage(const message_t &msg) = 0;
public:
   Core(unsigned messageBufferSize=MESSAGE_BUFFER_SIZE, unsigned debugLevel=DEBUG_NONE);
   virtual ~Core();
   int Start(message_t args="");
   bool StartTask(Task *t);
   bool KillTask(const compid_t &taskId);
   Task *childTask(const compid_t &taskId);
	bool SendIoMessage(const message_t &msg);
   bool DeliverIoMessage(message_t *msg);
	bool ReadIoMessage(const message_t &msg);
};

class Sys {
public:
	static uint64_t ticker();
	static int threads();
};

class ShellObject {
protected:
   virtual void monitor() {return;}
public:
   ShellObject() = default;
   virtual ~ShellObject() = default;
};

} // end_namespace AGENTS_Lib

#endif
