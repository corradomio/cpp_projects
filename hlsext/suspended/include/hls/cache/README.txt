stl::map
--------

    template<Key, Value, Compare, Allocator>
    class map

    key_type    -> Key
    value_type  -> Value
    size_type   -> size_t

    empty() -> bool
    size() -> size_type

    at(key) -> value | null
    operator[key] -> value | exception?

    begin/end, cbegin/cend

    clear()
    insert(key, value)
    erase(key)
    count(key)
    find(key)


Cache
-----

    CacheManager
    Cache
    Entry
    ExpiryPolicy


Usare un CacheManager (un singletine) che mantiene la gestione di TUTTE le cache del sistema.
Questo permette di assegnare un nome ad una cache del tipo

    name1.name2.name2

e configurare le cache sullo stile di log4j

    CacheManager.configure(file)
        carica la configurazione delle cache

    CacheManager.contains(name)
        controlla se una cache con quel nome esiste

    CacheManager.create(name, properties) -> Cache
        crea la cache con le properties specificate

    CacheManager.get(name) -> Cache
        se non esiste, la crea usando la configurazione

    CacheManager.delete(name)
        rimuove la cache

    CacheManager.delete()
        rimuove tutte le cache (ma non la configurazione)

    CacheManager.listCaches() -> list<Cache>
        ritorna la lista delle cache


Una cache deve necessariamente avere dei limiti:

    - numero massimo di oggetti
    - tempo in cui un oggetto puo' rimanere nella cache (indipendentemente
      dal numero di volte che e' stato utilizzato)
    - dopo quanto tempo di mancato utilizzo un oggetto puo' essere rimosso
    - memoria allocata?? (boh)

Specificare il numero massimo di oggetti, implica una strategia di rimozione
nel caso in cui la cache sia piena. Le possibili strategie sono:

    - gli oggetti che sono da troppo tempo nella cache
    - gli oggetto che sono da troppo tempo non utilizzati
    - gli oggetti che sono stati appena inseriti

La Cache viene creata con un ExpiryPolicy di default. In ogni caso, un oggetto
puo' specificare un ExpiryPolicy dedicata:

    Cache.contains(key) -> bool
    Cache.get(key) -> object | null
    Cache.add(key, object) -> Entry
    Cache.add(key, object, ExpiryPolicy) -> Entry
    Cache.remove(key) -> Entry
    Cache.clear()
    Cache.remove(pattern)

Startegie di deallocazione
--------------------------

    Random
    LRU         Last Recently Used
    MRU         Most Recently Used
    LFU         Last Frequently Used
    MFU         Most Frequently Used (???)
    LFU*        LFU, expires items with refcount == 1
    LFU-Age     LFU  time based
    LFU*-Age    LFU* time based
    Custom

    LRU/MRU: basta una linked-list
    LFU/MFU: serve una struttura dati in cui un oggetto puo' essere inserito in modo ordinato
             in base ad un indice (che puo' essere anche molto grande)

    LRU (Last Recenly Used): serve una linked list in cui, ogni volta che si usa un
    oggetto, viene messo in testa. Quindi si rimuove dalla cosa

    timeout: due possibili soluzioni
    1) l'oggetto puo' rimanere nella cache, e viene rimosso solo durante l'accesso
       (si controlla se e' scaduto)
       Ma questo puo' pregiudicare il funzionamento del LRU

    2) un thread, ogni N secondi, scandisce la cache e rimuove gli oggetti scaduti

    In ogni caso vale la pena usare un RWlock: read_lock se solo in lettura, write_lock
    in scrittura (quando viene inserito un nuovo oggetto o quando la cache viene
    scandita per la rimozione degli oggetti scaduti.

    Se la cache e' molto grande, potrebbe aver senso partizionara in modo che un'eventuale
    write_lock agisca SOLO su una partizione e non sull'intera cache.

    ATTENZIONE: LRU non e' la stessa cosa di LFU

    L'implementazione di LRU prevede una linked_list i cui, ogni volta che un oggetto viene
    usato, viene spostato in testa alla lista

    L'implementazione di LFU prevede un contatore usato per selezionare la lista su un vettore
    di liste indicizzate dal contatore stesso. Questa lista viene implementata com LRU.

    Supponiamo, ora, che ci siano DUE oggetti, uno usato 1000 volte e l'altro 0.
    Ora supponiamo che venga usato l'oggetto mai usato prima.
    Questo va a finire in testa alla LRU, ma rimane in una lista a bassa priorita' nella LFU.
    Con LRU, rimuovo l'oggetto usato molte volte, con LFU l'oggetto corretto.

Notifiche
---------

    - creazione della cache
    - inserimento di un oggetto nella cache
    - rimozione di un oggetto nella cache
    - rimozione di TUTTI gli oggetto (generare la notifica per ogni oggetto puo' essere
      complicato, se la cache ha milioni di oggetti)

References
----------

https://github.com/akashihi/stlcache
http://timday.bitbucket.org/lru.html

Java Cache
----------

JCS:        https://commons.apache.org/proper/commons-jcs/
Ehcache     http://www.ehcache.org/
JSR 107     https://jcp.org/en/jsr/detail?id=107