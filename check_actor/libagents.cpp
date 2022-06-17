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

#include "libagents.h"
#define cc(T,x) const_cast<T&>(x)

#include <stdlib.h>
#include <thread>
#include <iostream>

#define BROADCAST_RELAY_THREAD "|BROADCAST_RELAY_THREAD|"
namespace AGENTS_Lib {
// constexpr static const char *BROADCAST_RELAY_THREAD="|BROADCAST_RELAY_THREAD|";
   static unsigned debugLevel;
    static uint64_t startTime;
   class BroadcasterThread : public Thread {
      inline void onStarted() {}
   public:
      BroadcasterThread() : Thread(BROADCAST_RELAY_THREAD) {
         assert(!compid_t(BROADCAST_RELAY_THREAD).isValidId());
      }
      ~BroadcasterThread() {}
   };
}

using namespace AGENTS_Lib;

namespace _{} // message_t

message_t::message_t(const std::string &s) {
   qmsg.push_back(s);
}

message_t::message_t(const char *c) {
   qmsg.push_back((std::string)c);
}

message_t::message_t(double d) {
   qmsg.push_back(d);
}

message_t::message_t(int i) {
   qmsg.push_back(i);
}

message_t::message_t(unsigned u) {
   qmsg.push_back(u);
}

message_t::message_t(void *p) {
   qmsg.push_back(p);
}

message_t::message_t(const data_t &a) {
   qmsg.push_back(a);
}

void message_t::clear() {
   message_t clean_imsg;
   *this=clean_imsg;
}

//const message_t &message_t::operator=(const data_t &p) {
//   clear();
//   qmsg.push_back(p);
//   return *this;
//}

const data_t &message_t::operator[](unsigned i) const {
   force(i<qmsg.size());
   return qmsg[i];
}

data_t &message_t::operator[](unsigned i) {
   force(i<qmsg.size());
   return qmsg[i];
}

message_t &message_t::operator<<(const data_t &p) {
   qmsg.push_back(p);
   return *this;
}

message_t &message_t::insert(unsigned pos, const data_t &p) {
   qmsg.insert(qmsg.begin()+(pos<qmsg.size()? pos: qmsg.size()), p);
   return *this;
}

bool message_t::erase(unsigned pos) {
   if (pos<qmsg.size()) {
      qmsg.erase(qmsg.begin()+pos);
      return 1;
   }
   else return 0;
}

unsigned message_t::size() const {
   return qmsg.size();
}

std::string message_t::prettyPrint(std::string separator, bool decoration) const {
   std::string pp;
   for (unsigned i=0; i<qmsg.size(); i++) {
      if (i) pp+=separator;
      if (qmsg[i].is_string()) {
         if (decoration) pp+="$";
         pp+=qmsg[i].string();
      }
      else if (qmsg[i].is_number()) {
         double d=qmsg[i].number();
         if (decoration) pp+="#";
         if (d==(int)d) pp+=std::to_string((int)d);
         else pp+=std::to_string(d);
      }
      else if (qmsg[i].is_pointer()) {
         if (decoration) pp+="^";
         pp+=std::to_string((unsigned long long)qmsg[i].pointer());
      }
      else assert(0);
   }
   return pp;
}

bool message_t::to_string(std::string &smsg) const {
   force(qmsg.size()<=255); // messages longer than 255 cells cannot be serialized
   smsg="_"; smsg[0]=0;
   for (unsigned i=0; i<qmsg.size(); i++) {
      std::string qmsgi_str;
      if (qmsg[i].is_string()) {
         qmsgi_str=qmsg[i].string();
      }
      else if (qmsg[i].is_number()) {
         double d=qmsg[i].number();
         if (d==(int)d) qmsgi_str=std::to_string((int)d);
         else qmsgi_str=std::to_string(d);
      }
      else if (qmsg[i].is_pointer()) {
         qmsgi_str=std::to_string((unsigned long long)qmsg[i].pointer());
      }
      else assert(0);
      #define _strCellMaxSize 0x7fffffffUL // max string cell size set to 2GB-1
      force(qmsgi_str.size()<=_strCellMaxSize);
      uint32_t sz=(uint32_t)qmsgi_str.size();
      for (unsigned j=0, h=4; j<h; j++) smsg+=(char)(sz>>(j*8)&0xff);
      smsg+=(char)qmsg[i].t;
      smsg+=qmsgi_str;
      smsg[0]++;
   }
   force(smsg.size()<=_strCellMaxSize);
   return qmsg.size();
}

bool message_t::from_string(const std::string &smsg, int cap) {
   qmsg.clear();
   for (uint32_t i=0, p=1, h=4, l=smsg.size(), e; i<(smsg.size()? (unsigned char)smsg[0]: 0); i++, p+=e+h+1) {
      assert(p<=l-(h+1));
      e=0;
      for (unsigned j=0; j<h; j++) e+=smsg[p+j]<<(8*j);
      #define _messageCell smsg.substr(p+h+1,e)
      if (smsg[p+h]==data_t::ts)
         qmsg.push_back(_messageCell);
      else if (smsg[p+h]==data_t::td)
         qmsg.push_back(std::stod(_messageCell));
      else if (smsg[p+h]==data_t::tp)
         qmsg.push_back((void*)std::stoull(_messageCell));
      else assert(0);
   }
   for (int i=0; i<cap; i++) qmsg.push_back(""); // can append empty items so that i don't have to test for length in variable #args messages
   return smsg.size();
}


namespace _{} // MessageBuffer

uint64_t MessageBuffer::PushMessage(const message_t &msg, msgix_t msgIx) {
   static uint64_t msgCounter=0;
   static std::mutex editMsgCounter;
   uint64_t msgId;
   assert(msg.type!=message_t::undefined);
   std::lock_guard<std::recursive_mutex> lock(editMessageBuffer); // need recursive_mutex because this method can be called from [the same thread's] PopMessage()
   if (messages.size()<messageBufferSize) {
      if (msgIx) {
         msgId=msgIx.l;
      }
      else {
         std::lock_guard<std::mutex> lock(editMsgCounter);
         msgId=++msgCounter; // ensure msgId is never zero (start with ++0)
      }
      uint64_t now=time_ms();
      if (msg.sourceAgent) { // messages sent by Agent::SendMessage() or Agent::BroadcastMessage()
         assert((msg.sourceThreadId!="" && msg.sourceAgentId!=""));
         assert((msg.type==message_t::scheduled || msg.type==message_t::replicated) && msg.repeatCount>=1 && msg.deliverySchedule>=0 && msg.expireThreshold>=0);
         float schedule=msg.deliverySchedule;
         uint64_t delay=schedule;
         uint64_t dispersion=(schedule-delay)*delay*(float)(rand()%200-100)/110;
         msgIx.h=((msgIx? msgIx.h: now)+delay+dispersion)|(1ULL<<63); // bit63:=1 so that scheduled messages are popped after prioritized messages
         msgIx.l=msgId;
         std::lock_guard<std::mutex> lock(msg.sourceAgent->editMessageTracker);
         if (msg.type==message_t::scheduled && msg.sourceThread->enableCancelMessage) {
            msg.sourceAgent->messageTracker[msgId].first=this;
            msg.sourceAgent->messageTracker[msgId].second=msgIx;
         }
      }
      else { // messages sent by Core::SendIoMessage() or Core::SendMessage()
         assert(msg.type==message_t::prioritized && msg.repeatCount==0 && msg.deliverySchedule==0 && msg.expireThreshold==0 && msgIx==0);
         msgIx.h=now;   // this implicitly means bit63==0
         msgIx.l=msgId; // msgId is never zero (starts with 1)
      }
      assert(msgId!=0 && messages.count(msgIx)==0); // ### msgId rolls over after ~400 years for 1,000,000,000 messages/sec
      messages[msgIx]=msg;
//      { // ... test
//         static std::unordered_set<msgix_t> allPreviousIds;
//         if (allPreviousIds.count(msgIx)) std::cout<<msgIx.to_string()<<" ("<<msgId<<")"<<"\n";
//         allPreviousIds.insert(msgIx);
//      }
      return msgId;
   }
   else {
      assert(messages.size()==messageBufferSize);
      return 0;
   }
}

uint64_t MessageBuffer::PopMessage(message_t &msg) {
   msgix_t msgIx;
   uint64_t msgId, threshold;
   std::lock_guard<std::recursive_mutex> lock(editMessageBuffer);
   do {
      msg.clear();
      if (messages.size()==0) {
         return 0;
      }
      msgIx=messages.begin()->first;
      msgId=msgIx.l;
      threshold=time_ms()|(1ULL<<63);
      if (msgIx.h>threshold) {
         return 0;
      }
      msg=messages.begin()->second;
      messages.erase(msgIx);
      if (msg.sourceAgent) {
         assert(msg.type==message_t::scheduled || msg.type==message_t::replicated);
         std::lock_guard<std::mutex> lock(msg.sourceAgent->editMessageTracker);
         if (msg.sourceThread->enableCancelMessage) {
            if (msg.type==message_t::scheduled) {
               assert(msg.sourceAgent->messageTracker[msgId].first==this);
               assert(msg.sourceAgent->messageTracker[msgId].first->messages.count(msgIx)==0);
            }
            bool b=msg.sourceAgent->messageTracker.erase(msgId);
//          if (b)  if (msg.type!=message_t::scheduled) std::cout<<msgIx.to_string()<<"\n"; // ... test
            assert ((b && msg.type==message_t::scheduled) || (!b && msg.type==message_t::replicated));
         }
      }
      else {
         assert(msg.type==message_t::prioritized);
      }
      if (--msg.repeatCount>0) {
         uint64_t repeatMsgId=PushMessage(msg, msgIx); // editMessageBuffer is recursive_mutex so the buffer is, and stays, locked while PushMessage() is executed
         assert(repeatMsgId); // a message has just been popped, so there's always room to push [this] one back because no other thread could have pushed other messages (see comment above)
      }
   } while (msg.expireThreshold && (int)(threshold-(msgIx.h|(1ULL<<63)))>msg.expireThreshold); // this formula allows using expiration for prioritized messages, although prioritized messages have expireThreshold==0 in this version
   return msgId;
}


namespace _{} // Agent

uint32_t Agent::agentCounter=1; // don't have Id==0, just to play safe with zero values

Agent::Agent(const compid_t &_id) {
   force(_id.isValidId());
   force(agentCounter<(uint32_t)-1);
   id=_id;
   agentCounter++;
}

Agent::~Agent() {
   agentCounter--;
}

// 'schedule' parameter:
//    float format s.d where s=base delay in seconds, +/-0.d=relative dispersion
//    e.g. schedule=10.50 means 10 seconds with +/-50% dispersion, i.e. this designates a delay range of ~ 5s...15s

// the result of this function can be checked by the caller, and a failed Send() should be [repeatedly] rescheduled for a later time until it gets sent if the message has to have assured delivery
uint64_t Agent::SendMessage(const message_t &msg, const compid_t &destThreadId, const compid_t &destAgentId, float schedule, int repeat, int expire) {
   assert(parent);
   force(schedule>=0 && repeat>=1);
   force((destAgentId.isValidId() || destAgentId=="$" || destAgentId=="*") && (destThreadId.isValidId() || destThreadId=="$" || destThreadId=="*"));
   cc(message_t,msg).destThreadId=(destThreadId=="$"? parent->id : destThreadId);
   cc(message_t,msg).destAgentId=(destAgentId=="$"? id : destAgentId);
   cc(message_t,msg).sourceThreadId=parent->id;
   cc(message_t,msg).sourceAgentId=id;
   cc(message_t,msg).type=message_t::scheduled;
   cc(message_t,msg).repeatCount=repeat;
   cc(message_t,msg).deliverySchedule=schedule;
   cc(message_t,msg).expireThreshold=expire;
   cc(message_t,msg).sourceAgent=this;
   cc(message_t,msg).sourceThread=parent;
   Task *parentTask=parent->parentTask();
   std::lock_guard<std::recursive_mutex> lock(parentTask->editThreads);
   Thread *bufferingThread=parentTask->childThread(msg.destAgentId=="*"? BROADCAST_RELAY_THREAD : msg.destThreadId);
   uint64_t msgId=0;
   force(bufferingThread); // ... disallow losing messages sent to a nonexistent Thread, maybe i should make it a configuration option
   if (bufferingThread) msgId=bufferingThread->messageBuffer.PushMessage(msg);
   return msgId;
}

uint64_t Agent::BroadcastMessage(const message_t &msg, const compid_t &destThreadId, float schedule, int repeat, int expire) {
   return SendMessage(msg, destThreadId, "*", schedule, repeat, expire);
}

bool Agent::CancelMessage(uint64_t msgId) {
   force(parent->enableCancelMessage);
   bool r=0;
   std::lock_guard<std::mutex> lock(editMessageTracker);
   if (messageTracker.count(msgId)) {
      std::lock_guard<std::recursive_mutex> lock(messageTracker[msgId].first->editMessageBuffer);
      r=messageTracker[msgId].first->messages.erase(messageTracker[msgId].second);
      messageTracker.erase(msgId);
   }
   return r;
}

compid_t Agent::agentId() {
   return id;
}

Thread *Agent::parentThread() {
   return parent;
}


namespace _{} // Thread

Thread::Thread(const compid_t &_id, unsigned messageBufferSize) {
    messageBuffer.messageBufferSize=messageBufferSize;
   force(_id.isValidId() || _id==BROADCAST_RELAY_THREAD);
   id=_id;
}

Thread::~Thread() {
   Stop();
   std::lock_guard<std::recursive_mutex> lock(editAgents);
   while (registeredAgents.size()) {
      compid_t agent_id=registeredAgents.begin()->second->id; // need to use a temp variable because the value is passed by reference to Kill below and this expression becomes invalid during Kill
      force(KillAgent(agent_id));
   }
}

void Thread::Start() {
   assert(parent);
// assert(registeredAgents.size()); // cannot assert because i want to allow starting a Thread even before inserting the very first Agent in the Thread
   assert (!isRunning);
   runSwitch=1;
   isRunning=1;
   messageLoop();
   isRunning=0;
}

void Thread::Stop() {
   assert(isRunning);
   runSwitch=0;
   while (isRunning) sleep_ms(OS_TICK);
   sleep_ms(2*OS_TICK); // make sure the Thread's Start() method completes its return
}

void Thread::messageLoop() {
   message_t msg;
   while (runSwitch) {
      while (runSwitch && messageBuffer.PopMessage(msg)) {
         DispatchMessage(msg);
      }
      sleep_ms(OS_TICK);
   }
}

int Thread::DispatchMessage(message_t &msg) {
   if (msg.destAgentId!="*") { // targeted message
      assert(id!=BROADCAST_RELAY_THREAD && msg.destThreadId==id);
      std::lock_guard<std::recursive_mutex> lock(editAgents);
      Agent *destAgent=childAgent(msg.destAgentId);
      force(destAgent); // ... disallow losing messages sent to a nonexistent Agent, maybe i should make it a configuration option
      if (destAgent) {
         force(destAgent->started); // ... disallow losing messages sent to a not-yet-started Agent
         if (destAgent->started) {
            force(destAgent->onMessage(msg, msg.sourceThreadId, msg.sourceAgentId));
         }
         return 1;
      }
      else
         return 0;
   }
   else { // broadcasted message
      assert(id==BROADCAST_RELAY_THREAD && msg.destThreadId!=id);
      assert(msg.type=message_t::scheduled);
      // ??? better to prioritize sending the messages to other threads first?
      compid_t destThreadId=msg.destThreadId;
      msg.deliverySchedule=0;
      msg.repeatCount=1;
      msg.type=message_t::replicated;
      std::lock_guard<std::mutex> lock(parent->editMessageRoutingTable);
      auto matches0=parent->messageRoutingTable.equal_range(Task::messageRoutingTableIndex("*", msg.sourceThreadId, msg.sourceAgentId));
      auto matches1=parent->messageRoutingTable.equal_range(Task::messageRoutingTableIndex(msg[0], msg.sourceThreadId, msg.sourceAgentId));
      std::deque<decltype(matches0)> matches({matches0, matches1});
      for (auto m: matches) {
         for (auto i=m.first; i!=m.second; i++) {
            msg.destThreadId=std::get<0>(i->second);
            msg.destAgentId=std::get<1>(i->second);
            if (msg.destThreadId==destThreadId || destThreadId=="*") {
               while (!parent->RelayMessage(msg)) sleep_ms(OS_TICK);
            }
         }
      }
      return 1;
   }
}

bool Thread::StartAgent(Agent *a) {
   force(parentTask() && parentTask()->core);
   editAgents.lock();
   if (!registeredAgents.count(a->id.to_unique_str())) {
      a->parent=this;
        a->core=parent->core;
      registeredAgents[a->id.to_unique_str()]=a;
      editAgents.unlock(); // must place this unlock() before onStarted because onStarted may StartAgent(new Agent), and thus this function could otherwise potentially get blocked
      a->started=1;
      a->onStarted();
      return 1;
   }
   else {
      editAgents.unlock();
      return 0;
   }
}

bool Thread::KillAgent(const compid_t &agentId) {
   std::lock_guard<std::recursive_mutex> lock(editAgents);
   parentTask()->RemoveAgentRouting(threadId(), agentId);
   bool r=0;
   if (registeredAgents.count(agentId.to_unique_str())) {
      delete(registeredAgents[agentId.to_unique_str()]);
      registeredAgents.erase(agentId.to_unique_str());
      r=1;
   }
   return r;
}

Agent *Thread::operator[](const compid_t &agentId) {
   std::lock_guard<std::recursive_mutex> lock(editAgents);
   if (registeredAgents.count(agentId.to_unique_str()))
      return registeredAgents[agentId.to_unique_str()];
   else
      return nullptr;
}

Agent *Thread::childAgent(const compid_t &agentId) {
   return (*this)[agentId];
}

compid_t Thread::threadId() {
   return id;
}

Task *Thread::parentTask() {
   return parent;
}


namespace _{} // Task

Task::Task(const compid_t &_id) {
   force(_id.isValidId());
   id=_id;
   assert(StartThread(new BroadcasterThread));
}

Task::~Task() {
   while (registeredThreads.size()) {
      compid_t thread_id=registeredThreads.begin()->second->id; // need to use a temp variable because the value is passed by reference to Kill below and this expression becomes invalid during Kill
      force(KillThread(thread_id));
   }
}

bool Task::RelayMessage(const message_t &msg) {
   // no (double-)checks for destXXXId.isValidId()'s in this method because special messages (i.e. with invalid destXXXId's) may also be relayed (in this or future versions of the lib)
   std::lock_guard<std::recursive_mutex> lock(editThreads);
   Thread *destThread=childThread(msg.destThreadId);
   force(destThread); // ... disallow losing messages sent to a nonexistent Thread, maybe i should make it a configuration option
   if (destThread)
      return (bool)destThread->messageBuffer.PushMessage(msg);
   else
      return 0;
}

bool Task::StartThread(Thread *t) {
   force(core || t->id==BROADCAST_RELAY_THREAD);
   editThreads.lock();
   if (!registeredThreads.count(t->id.to_unique_str())) {
      t->parent=this;
        t->core=parent;
      registeredThreads[t->id.to_unique_str()]=t;
      editThreads.unlock(); // must place this unlock() before onStarted because onStarted may StartThread(new Thread), and thus this function could otherwise potentially get blocked
      new std::thread(&Thread::Start, t);
      while (!t->isRunning) sleep_ms(OS_TICK);
      sleep_ms(2*OS_TICK); // make sure the Thread's Start() method enters the message loop
      t->onStarted();
      return 1;
   }
   else {
      editThreads.unlock();
      return 0;
   }
}

bool Task::KillThread(const compid_t &threadId) {
   std::lock_guard<std::recursive_mutex> lock(editThreads);
   bool r=0;
   if (registeredThreads.count(threadId.to_unique_str())) {
      delete(registeredThreads[threadId.to_unique_str()]);
      registeredThreads.erase(threadId.to_unique_str());
      r=1;
   }
   return r;
}

std::string Task::messageRoutingTableIndex(const data_t &messageName, const compid_t &sourceThreadId, const compid_t &sourceAgentId) {
   return messageName.to_unique_str()+"|"+sourceThreadId.to_unique_str()+"|"+sourceAgentId.to_unique_str();
}
// ??? maybe make the MessageRouter a class?
bool Task::AddBroadcastSubscription(const data_t &messageName, const compid_t &sourceThreadId, const compid_t &sourceAgentId, const compid_t &destThreadId, const compid_t &destAgentId) {
   Thread *sourceThread=childThread(sourceThreadId);
   Thread *destThread=childThread(destThreadId);
   bool sourceDestExist = (sourceThread &&
                           sourceThread->childAgent(sourceAgentId) &&
                           destThread &&
                           destThread->childAgent(destAgentId));
   bool subscriptionExists=RemoveBroadcastSubscription(messageName, sourceThreadId, sourceAgentId, destThreadId, destAgentId);
   force(sourceDestExist); // ... disallow routing messages to/from a nonexistent thread.agent, maybe i should make it a configuration option?
   force(!RemoveBroadcastSubscription(messageName, sourceThreadId, sourceAgentId, destThreadId, destAgentId));
   std::string routeSourceHandle=messageRoutingTableIndex(messageName, sourceThreadId, sourceAgentId);
   std::lock_guard<std::mutex> lock(editMessageRoutingTable);
   messageRoutingTable.insert(std::pair<std::string,std::tuple<compid_t,compid_t>>
                              (routeSourceHandle, std::tuple<compid_t,compid_t>(destThreadId, destAgentId))
                              );
   return subscriptionExists;
}

bool Task::RemoveBroadcastSubscription(const data_t &messageName, const compid_t &sourceThreadId, const compid_t &sourceAgentId, const compid_t &destThreadId, const compid_t &destAgentId) {
   std::string routeSourceHandle=messageRoutingTableIndex(messageName, sourceThreadId, sourceAgentId);
   auto target=std::tuple<compid_t,compid_t>(destThreadId, destAgentId);
   std::lock_guard<std::mutex> lock(editMessageRoutingTable);
   auto matches=messageRoutingTable.equal_range(routeSourceHandle);
   for (auto i=matches.first; i!=matches.second; i++) {
      if (i->second==target) {
         assert(i->first==routeSourceHandle);
         messageRoutingTable.erase(i);
         return 1;
      }
   }
   return 0;
}

int Task::RemoveAgentRouting(const compid_t &threadId, const compid_t &agentId) {
   int r=0;
   std::string agent=threadId.to_unique_str()+"|"+agentId.to_unique_str();
   std::lock_guard<std::mutex> lock(editMessageRoutingTable);
   for (auto index=messageRoutingTable.begin(); index!=messageRoutingTable.end();) {
      std::string source=index->first.substr(1+index->first.rfind('|', index->first.rfind('|')-1)); // substring after the message name
      std::string dest=std::get<0>(index->second).to_unique_str()+"|"+std::get<1>(index->second).to_unique_str();
      if (source==agent || dest==agent) {
         messageRoutingTable.erase(index);
         index=messageRoutingTable.begin();
         r++;
      }
      else index++;
   }
   return r;
}

Thread *Task::operator[](const compid_t &threadId) {
   std::lock_guard<std::recursive_mutex> lock(editThreads);
   if (registeredThreads.count(threadId.to_unique_str()))
      return registeredThreads[threadId.to_unique_str()];
   else
      return nullptr;
}

Thread *Task::childThread(const compid_t &threadId) {
   return (*this)[threadId];
}

compid_t Task::taskId() {
   return id;
}

Core* Task::parentCore() {
    return parent;
}


namespace _{} // Core

Core::Core(unsigned messageBufferSize, unsigned debugLevel) : intercom(this) {
   // 'unsigned long long' must be able to hold a 'uintptr_t', i.e. "some_void_pointer==(void*)(unsigned long long)some_void_pointer" holds true
   // rationale: have to use 'unsigned long long' because i need to convert a pointer to/from a numeric string, and std::string offers conversions to/from 'unsinged long long' but not to/from 'uintptr_t'
    force(debugLevel==DEBUG_NONE); // debugLevel NYI
    force(sizeof(unsigned long long)>=sizeof(uintptr_t));
    AGENTS_Lib::debugLevel=debugLevel;
    intercom.rxBuffer.messageBufferSize=messageBufferSize;
    intercom.txBuffer.messageBufferSize=messageBufferSize;
}

Core::~Core() {
/*
   while (registeredTasks.size()) { // >>> KillTask() NYI
      compid_t task_id=registeredTasks.begin()->second->id; // need to use a temp variable because the value is passed by reference to Kill below and this expression becomes invalid during Kill
      force(KillTask(task_id));
   }
   sleep_ms(2*OS_TICK); // make sure the inputMonitor() completes its return
*/
}

Core::Intercom::Intercom(Core *parent) {
   this->core=parent;
   inputMonitorIsRunning=0;
   inputMonitorRunSwitch=1;
   new std::thread(&Core::Intercom::inputMonitor, this);
   while (!inputMonitorIsRunning) sleep_ms(OS_TICK);
   sleep_ms(2*OS_TICK); // make sure the inputMonitor() started
}

Core::Intercom::~Intercom() {
   inputMonitorRunSwitch=0;
   while (inputMonitorIsRunning) sleep_ms(OS_TICK);
   sleep_ms(2*OS_TICK); // make sure the inputMonitor() stopped
}

void Core::Intercom::inputMonitor() {
   message_t msg;
   inputMonitorIsRunning=1;
   while (inputMonitorRunSwitch) {
      if (rxBuffer.PopMessage(msg)) force(core->onIoMessage(msg));
      else sleep_ms(OS_TICK);
   }
   inputMonitorIsRunning=0;
}

bool Core::Intercom::PushMessage(const message_t &msg) {
   cc(message_t,msg).sourceTaskId=cc(message_t,msg).sourceThreadId=cc(message_t,msg).sourceAgentId="";
   cc(message_t,msg).destThreadId=cc(message_t,msg).destAgentId="";
   cc(message_t,msg).type=message_t::prioritized;
   cc(message_t,msg).deliverySchedule=cc(message_t,msg).repeatCount=cc(message_t,msg).expireThreshold=0;
   return (bool)rxBuffer.PushMessage(msg);
}

bool Core::Intercom::PopMessage(message_t &msg) {
   return (bool)txBuffer.PopMessage(msg);
}

int Core::Start(message_t args) {
    startTime=time_ms();
   srand((int) startTime);
   return onStarted(args);
}

Task *Core::operator[](const compid_t &taskId) {
   std::lock_guard<std::recursive_mutex> lock(editTasks);
   if (registeredTasks.count(taskId.to_unique_str()))
      return registeredTasks[taskId.to_unique_str()];
   else
      return nullptr;
}

Task *Core::childTask(const compid_t &taskId) {
   return (*this)[taskId];
}

bool Core::StartTask(Task *t) {
   editTasks.lock();
   if (!registeredTasks.count(t->id.to_unique_str())) {
      t->parent=this;
        t->core=this;
      registeredTasks[t->id.to_unique_str()]=t;
      editTasks.unlock(); // must place this unlock() before onStarted because onStarted may StartTask(new Task), and thus this function could otherwise potentially get blocked
      t->onStarted();
      return 1;
   }
   else {
      editTasks.unlock();
      return 0;
   }
}

bool Core::SendMessage(const message_t &msg, const compid_t &destTaskId, const compid_t &destThreadId, const compid_t &destAgentId) {
   force(destTaskId.isValidId() && destThreadId.isValidId() && destAgentId.isValidId());
   cc(message_t,msg).sourceThreadId=cc(message_t,msg).sourceAgentId="";
   cc(message_t,msg).destThreadId=destThreadId;
   cc(message_t,msg).destAgentId=destAgentId;
   cc(message_t,msg).type=message_t::prioritized;
   cc(message_t,msg).deliverySchedule=cc(message_t,msg).repeatCount=cc(message_t,msg).expireThreshold=0;
   std::lock_guard<std::recursive_mutex> lock(editTasks);
   Task *destTask=childTask(destTaskId);
   force(destTask); // ... disallow losing messages sent to a nonexistent Task, maybe i should make it a configuration option
   if (destTask)
      return destTask->RelayMessage(msg);
   else
      return 0;
}

bool Core::SendIoMessage(const message_t &msg) {
   cc(message_t,msg).sourceTaskId=cc(message_t,msg).sourceThreadId=cc(message_t,msg).sourceAgentId="";
   cc(message_t,msg).destThreadId=cc(message_t,msg).destAgentId="";
   cc(message_t,msg).type=message_t::prioritized;
   cc(message_t,msg).deliverySchedule=cc(message_t,msg).repeatCount=cc(message_t,msg).expireThreshold=0;
   return (bool)intercom.txBuffer.PushMessage(msg);
}

bool Core::DeliverIoMessage(message_t *msg) {
   return intercom.PopMessage(*msg);
}

bool Core::ReadIoMessage(const message_t &msg) {
   return intercom.PushMessage(msg);
}

namespace _{} // Sys

uint64_t Sys::ticker() {
    return time_ms()-startTime;
}

int Sys::threads() {
   return std::thread::hardware_concurrency();
}
