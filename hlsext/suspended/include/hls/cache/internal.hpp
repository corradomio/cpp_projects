//
// Created by Corrado Mio on 20/03/2016.
//

#ifndef HLS_COLLECTION_LINKEDLIST_HPP
#define HLS_COLLECTION_LINKEDLIST_HPP

namespace hls {
namespace collection {
namespace internal {

    template<typename Value>
    struct node {
        long  hash;
        long  count;
        node* prev;
        node* next;
        Value value;
    };


    template<typename Value>
    class linked_list
    {
        typedef node<Value> node_t;

        node_t* free;
        node_t* head;
        node_t* tail;

        long count;
        long nfree;

    public:
        linked_list(): free(nullptr), head(nullptr), tail(nullptr) { }
        ~linked_list() {
            node_t *n;
            while(free) { n = free; free = free->next; delete n; }
            while(head) { n = head; head = head->next; delete n; }
        }

        node_t* add(const Value& value) {
            node_t* n = new_node();
            n->value = value;
            add_node(n);
            return n;
        }

        void remove(node_t* n) {
            remove_node(n);
            free_node(n);
        }

        void front(node_t* n) {
            remove_node(n);
            add_node(n);
        }

        node_t* select() { return tail; }

    private:

        node_t* new_node() {
            node_t* n = free;
            if (n)
                free = free->next, --nfree;
            else
                n = new node<Value>;

            return n;
        }

        void free_node(node_t* n)
        {
            n->value = Value();
            n->next = free;
            free = n;

            ++nfree;
        }

        void add_node(node_t* n)
        {
            n->next = head;
            n->prev = nullptr;
            head = n;
            if (!tail) tail = n;

            ++count;
        }

        void remove_node(node_t* n)
        {
            if (n->prev)
                n->prev->next = n->next;
            else
                head = n->next;

            if (n->next)
                n->next->prev = n->prev;
            else
                tail = n->prev;

            --count;
        }

    };

    template<typename Value>
    class bucket_list
    {
    protected:
        typedef node<Value> node_t;

        struct entry_t {
            node_t* head;
            node_t* tail;
            long count;
        };

        entry_t*bucket;
        node_t* free;
        long    count;

    public:
        bucket_list() {
            bucket = new entry_t[32];
            for(int i=0; i<32; ++i) bucket[i] = nullptr;
        }
        ~bucket_list() {
            for(int i=0; i<32; ++i) {
                node_t *t, *n = bucket[i].head;
                while(n) { t = n->next; delete n; n = t; }
            }
            delete[] bucket;
        }

    protected:

        node_t* new_node() {
            node_t* n = free;
            if (n)
                free = free->next;
            else
                n = new node<Value>;

            n->count = 0;
            n->hash = 0;
            n->next = n->prev = nullptr;
            n->value = Value();
            return n;
        }

        void add_node(node_t* n, long selector){
            int i = indexof(selector);

            n->next = bucket[i].head;
            if (!bucket[i].tail)
                bucket[i].tail = n;
        }

        void remove_node(node_t* n, long selector) {
            int i = indexof(selector);

            if (n->prev)
                n->prev->next = n->next;
            else
                bucket[i].head = n->next;

            if (n->next)
                n->next->prev = n->prev;
            else
                bucket[i].tail = n->prev;

            n->next = n->prev = nullptr;
        }

        void free_node(node_t* n, long selector) {
            remove_node(n, selector);

            n->value = Value();
            n->next = free;
            free =  n;
        }

        int indexof(long selector) {
            long limit = 1;
            int index = 0;
            while(selector > limit) {
                ++index;
                limit += limit;
            }
            return index;
        }

    };


    template<typename Value>
    class cbucket_list : public bucket_list<Value>
    {
    public:
        node_t* add(const Value& value) {
            node_t* n = new_node();
            n->value = value;
            add_node(n, 0);
            return n;
        }

        void remove(node_t* n) {
            free_node(n, n->count);
        }

        void incr(node_t* n) {
            remove_node(n, n->count);
            n->count++;
            add_node(n, n->count);
        }

        node_t* select() {
            for(int i=0; i<32; ++i)
                if(bucket[i].tail)
                    return bucket[i].tail;
            return nullptr;
        }

    };

    template<typename Value>
    class hbucket_list : public bucket_list<Value>
    {
    public:
        node_t* add(long hash, const Value& value) {
            node_t* n = new_node();
            n->hash = hash;
            n->value = value;
            add_node(n, hash);
            return n;
        }

        void remove(node_t* n) {
            free_node(n, n->hash);
        }

        node_t* select() {
            for(int i=0; i<32; ++i)
                if(bucket[i].head)
                    return bucket[i].head;
            return nullptr;
        }

    };


}}}

#endif //HLS_COLLECTION_LINKEDLIST_HPP
