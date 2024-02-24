//
// Created by Corrado Mio on 11/02/2024.
//

#ifndef CHECK_STRUCTURES_LIST_H
#define CHECK_STRUCTURES_LIST_H

namespace stdx {

    template<typename T>
    struct list_t {
        struct node_t {
            node_t *next;
            T data;

            node_t() { }
            node_t(const T& e): next(nullptr), data(e) { }
        };

        size_t n;
        node_t *head;

        /// Constructor
        explicit list_t(): head(nullptr), n(0) { }
        /// Destructor
        ~list_t() { clear(); }

        [[nodiscard]] bool  empty() const { return n == 0; }
        [[nodiscard]] size_t size() const { return n; }
        [[nodiscard]] node_t* first() const { return head; }

        /// Add a new element at the head of the list and return its pointer
        node_t* add(const T& e) {
            auto *c = new node_t(e);
            if (head == nullptr) {
                head = c;
            } else {
                c->next = head;
                head = c;
            }
            ++n;
            return c;
        }

        void clear() {
            node_t *c = head;
            while (c != nullptr) {
                node_t *n = c->next;
                delete c;
                c = n;
            }
            n = 0;
            head = nullptr;
        }
    };

}

#endif //CHECK_STRUCTURES_LIST_H
