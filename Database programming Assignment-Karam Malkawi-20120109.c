#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <openssl/evp.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

#define PAGE_SIZE 4096 // 4 KB
#define BUFFER_POOL_SIZE 100 // Number of pages in the buffer pool
#define HASH_TABLE_SIZE 1000 // Size of the hash table
#define B_TREE_ORDER 4 // Order of the B-tree
#define INITIAL_CAPACITY 100 // Initial capacity for tables

// Log entry structure
typedef struct {
    int transaction_id;
    char operation_type; // 'I' for insert, 'U' for update, 'D' for delete
    int page_id;
    char old_data[PAGE_SIZE];
    char new_data[PAGE_SIZE];
    time_t timestamp; // Add timestamp
    char table_name[50]; // Add table name
    int record_id; // Add record ID
    char old_value[100]; // Add old value
    char new_value[100]; // Add new value
} LogEntry;

// Task structure for thread pool
typedef struct {
    void (*task_function)(void *); // Function to execute
    void *task_data;               // Data for the task
} Task;

typedef struct {
    Task *tasks;
    int queue_size;
    int task_count;
    int head;
    int tail;
    pthread_mutex_t queue_lock;
    pthread_cond_t task_available; // Replace sem_t with pthread_cond_t
    pthread_t *threads;
    int num_threads;
    bool shutdown;
} ThreadPool;

typedef struct BufferPage {
    int page_id;
    char data[PAGE_SIZE];
    bool is_dirty;
    int pin_count;
    bool is_locked;
    char lock_mode;
    int s_lock_count; // Shared lock count
    pthread_mutex_t lock_mutex; // Mutex for lock management
    pthread_cond_t lock_cond; // Condition variable for lock management
    time_t last_used; // Add this field for LRU
    struct BufferPage *next;
} BufferPage;

// Structure to pass arguments to the thread function
typedef struct {
    BufferPage *page;
    char lock_mode;
} ThreadArg;

// Buffer pool structure
typedef struct {
    BufferPage pages[BUFFER_POOL_SIZE];
    int num_pages;
    pthread_mutex_t lock; // Mutex for thread safety
} BufferPool;

// Transaction structure
typedef struct {
    int transaction_id;
    BufferPage *locked_pages; // List of pages locked by this transaction
} Transaction;

// Hash entry structure
typedef struct HashEntry {
    char key[100];
    int index;
    struct HashEntry *next;
} HashEntry;

// Hash table structure
typedef struct HashTable {
    HashEntry **buckets;
    int size;
    pthread_mutex_t *bucket_locks; // Array of mutexes, one per bucket
} HashTable;

// Index entry structure
typedef struct {
    int id; // Record ID
    int index; // Main table index
} IndexEntry;

// B-tree node structure
typedef struct BTreeNode {
    int num_keys; // Number of keys in the node
    IndexEntry keys[B_TREE_ORDER - 1]; // Array of keys
    struct BTreeNode *children[B_TREE_ORDER]; // Array of child pointers
    bool is_leaf; // Whether the node is a leaf
} BTreeNode;

// B-tree structure
typedef struct {
    BTreeNode *root; // Root of the B-tree
} BTree;

// Memory block structure
typedef struct MemoryBlock {
    void *data;
    size_t size;
    struct MemoryBlock *next;
} MemoryBlock;

// Memory manager structure
typedef struct MemoryManager {
    MemoryBlock *head;
} MemoryManager;

// Index table structure
typedef struct {
    BTree btree; // B-tree for indexing
    HashTable email_hash_table; // Hash table for email lookups
    HashTable phone_hash_table; // Hash table for phone lookups
    IndexEntry *entries; // Array of index entries
    int capacity; // Capacity of the index table
    int count; // Number of entries in the index table
} IndexTable;

// Table structure
typedef struct {
    char name[50];
    void *data;
    size_t element_size;
    int capacity;
    int count;
    bool *in_use;
    int last_id;
    IndexTable index_table;
    BufferPool buffer_pool;
    MemoryManager memory_manager;
    pthread_mutex_t lock; // Mutex for thread-safe access
} Table;


typedef struct {
    Table *table;
    int id;
    void *result;
    pthread_mutex_t *result_mutex;
} SelectTaskData;

// Faculty structure
typedef struct {
    int id;
    char name[100];
    char dean[100];
} Faculty;

// Department structure
typedef struct {
    int id;
    char name[100];
    int faculty_id;
} Department;

// Student structure
typedef struct {
    int id;
    char first_name[50];
    char last_name[50];
    char email[100];
    char phone[15];
    char password[65];
    int department_id;
    char date_of_birth[11];
    char age[4];
    float passed_hours;
    char country[100];
    char city[100];
    char street[100];
} Student;

// Instructor structure
typedef struct {
    int id;
    char first_name[50];
    char last_name[50];
    char email[100];
    char phone[15];
    char password[65];
    char SSN[12];
    char date_of_birth[11];
    char age[4];
    float salary;
    char country[100];
    char city[100];
    char street[100];
    int department_id;
} Instructor;

// Course structure
typedef struct {
    int id;
    char title[50];
    char code[50];
    char active_status[50];
    float hours;
    int department_id;
} Course;

// InstructorCourses structure
typedef struct {
    int id;
    int courses_id;
    int instructor_id;
} InstructorCourses;

// StudentCourses structure
typedef struct {
    int id;
    int courses_id;
    int student_id;
} StudentCourses;

// Function Declarations
void initialize_table(Table *table, const char *name, size_t element_size, int capacity);
void initialize_hash_table(HashTable *table, int size);
void initialize_index(const char *filename);
void load_all_tables(Table *faculty_table, Table *department_table, Table *student_table, Table *instructor_table,
                     Table *course_table, Table *instructor_courses_table, Table *student_courses_table);
void main_menu(Table *faculty_table, Table *department_table, Table *student_table, Table *instructor_table,
        Table *course_table, Table *instructor_courses_table, Table *student_courses_table, FILE *log_file);

// B-Tree Functions
BTreeNode *create_btree_node(bool is_leaf);
IndexEntry *btree_search(BTreeNode *root, int id);
void btree_insert(BTree *tree, IndexEntry entry);

// Hash Table Functions
unsigned int hash_function(const char *key);
void hash_table_insert(HashTable *table, const char *key, int index);
int hash_table_search(HashTable *table, const char *key);
void resize_hash_table(HashTable *table, int new_size);
void free_hash_table(HashTable *table);

// Buffer Pool Functions
BufferPage *find_page(BufferPool *buffer_pool, int page_id);

// Transaction Functions
bool is_lock_compatible(char requested_lock, char existing_lock);
bool request_lock(BufferPage *page, char lock_mode);
void release_lock(BufferPage *page, char lock_mode);
int generate_transaction_id();
Transaction *begin_transaction();
void commit_transaction(Transaction *transaction);
void write_log_entry(FILE *log_file, LogEntry *entry);
void recover_database(FILE *log_file, BufferPool *buffer_pool);

// Parallel Processing Functions
void insert_task(void *data);
bool insert_into_table(Table *table, void *element, int *id, const char *filename, Table *related_table, FILE *log_file);
void parallel_insert(ThreadPool *pool, Table *table, void *element, const char *filename, Table *related_table);

void delete_from_table(Table *table, int id, const char *filename, FILE *log_file);

// Add these prototypes at the beginning of the file
void btree_delete_recursive(BTreeNode *node, int id);
int find_key_index(BTreeNode *node, int id);
void remove_from_leaf(BTreeNode *node, int idx);
void remove_from_non_leaf(BTreeNode *node, int idx);
void fill_child(BTreeNode *node, int idx);
IndexEntry get_predecessor(BTreeNode *node, int idx);
IndexEntry get_successor(BTreeNode *node, int idx);
void borrow_from_prev(BTreeNode *node, int idx);
void borrow_from_next(BTreeNode *node, int idx);
void merge_children(BTreeNode *node, int idx);

void hash_table_delete(HashTable *table, const char *key);
void write_page(BufferPool *buffer_pool, int page_id, const char *filename);

FILE *open_log_file(const char *filename);


Transaction *current_transaction = NULL;

// Main Function
int main() {
    // Open the log file
    FILE *log_file = open_log_file("database.log");
    if (!log_file) {
        return 1; // Exit if the log file cannot be opened
    }

    // Initialize tables and buffer pool
    Table faculty_table, department_table, student_table, instructor_table, course_table, instructor_courses_table, student_courses_table;
    initialize_table(&faculty_table, "Faculty", sizeof(Faculty), INITIAL_CAPACITY);
    initialize_table(&department_table, "Department", sizeof(Department), INITIAL_CAPACITY);
    initialize_table(&student_table, "Student", sizeof(Student), INITIAL_CAPACITY);
    initialize_table(&instructor_table, "Instructor", sizeof(Instructor), INITIAL_CAPACITY);
    initialize_table(&course_table, "Course", sizeof(Course), INITIAL_CAPACITY);
    initialize_table(&instructor_courses_table, "InstructorCourses", sizeof(InstructorCourses), INITIAL_CAPACITY);
    initialize_table(&student_courses_table, "StudentCourses", sizeof(StudentCourses), INITIAL_CAPACITY);

    // Load all tables from files
    load_all_tables(&faculty_table, &department_table, &student_table, &instructor_table, &course_table, &instructor_courses_table, &student_courses_table);

    // Start a new transaction
    current_transaction = begin_transaction();

    // Run the main menu
    main_menu(&faculty_table, &department_table, &student_table, &instructor_table, &course_table, &instructor_courses_table, &student_courses_table, log_file);

    // Commit the transaction
    commit_transaction(current_transaction);

    // Close the log file
    fclose(log_file);

    return 0;
}

// 
FILE *open_log_file(const char *filename) {
    FILE *log_file = fopen(filename, "a"); // Open in append mode
    if (!log_file) {
        perror("Failed to open log file");
        return NULL;
    }
    return log_file;
}

void write_log_entry(FILE *log_file, LogEntry *entry) {
    if (!log_file || !entry) {
        return;
    }

    // Convert timestamp to a human-readable format
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&entry->timestamp));

    // Write the log entry in CSV format
    fprintf(log_file, "%s,%c,%s,%d,%s,%s,%d\n",
            timestamp,
            entry->operation_type,
            entry->table_name,
            entry->record_id,
            entry->old_value,
            entry->new_value,
            entry->transaction_id);
}

void log_insert(FILE *log_file, const char *table_name, int record_id, const char *new_value, int transaction_id) {
    LogEntry entry;
    entry.timestamp = time(NULL);
    entry.operation_type = 'I';
    strncpy(entry.table_name, table_name, sizeof(entry.table_name));
    entry.record_id = record_id;
    strncpy(entry.new_value, new_value, sizeof(entry.new_value));
    entry.old_value[0] = '\0'; // No old value for inserts
    entry.transaction_id = transaction_id;

    write_log_entry(log_file, &entry);
}

void log_update(FILE *log_file, const char *table_name, int record_id, const char *old_value, const char *new_value, int transaction_id) {
    LogEntry entry;
    entry.timestamp = time(NULL);
    entry.operation_type = 'U';
    strncpy(entry.table_name, table_name, sizeof(entry.table_name));
    entry.record_id = record_id;
    strncpy(entry.old_value, old_value, sizeof(entry.old_value));
    strncpy(entry.new_value, new_value, sizeof(entry.new_value));
    entry.transaction_id = transaction_id;

    write_log_entry(log_file, &entry);
}

void log_delete(FILE *log_file, const char *table_name, int record_id, const char *old_value, int transaction_id) {
    LogEntry entry;
    entry.timestamp = time(NULL);
    entry.operation_type = 'D';
    strncpy(entry.table_name, table_name, sizeof(entry.table_name));
    entry.record_id = record_id;
    strncpy(entry.old_value, old_value, sizeof(entry.old_value));
    entry.new_value[0] = '\0'; // No new value for deletes
    entry.transaction_id = transaction_id;

    write_log_entry(log_file, &entry);
}

// Worker thread function
void *worker_thread(void *arg) {
    ThreadPool *pool = (ThreadPool *)arg;
    while (1) {
        pthread_mutex_lock(&pool->queue_lock);

        // Check for shutdown signal
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->queue_lock);
            break;
        }

        // Wait for a task to be available
        while (pool->task_count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->task_available, &pool->queue_lock);
        }

        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->queue_lock);
            break;
        }

        // Get the next task
        Task task = pool->tasks[pool->head];
        pool->head = (pool->head + 1) % pool->queue_size;
        pool->task_count--;

        pthread_mutex_unlock(&pool->queue_lock);

        // Execute the task
        if (task.task_function) {
            task.task_function(task.task_data);
        }
    }
    return NULL;
}

// Initialize thread pool
void initialize_thread_pool(ThreadPool *pool, int num_threads, int queue_size) {
    pool->tasks = (Task *)malloc(queue_size * sizeof(Task));
    pool->queue_size = queue_size;
    pool->task_count = 0;
    pool->head = 0;
    pool->tail = 0;
    pthread_mutex_init(&pool->queue_lock, NULL);
    pthread_cond_init(&pool->task_available, NULL); // Initialize condition variable
    pool->threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    pool->num_threads = num_threads;
    pool->shutdown = false;

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }
}

// Add task to thread pool
void add_task(ThreadPool *pool, void (*task_function)(void *), void *task_data) {
    pthread_mutex_lock(&pool->queue_lock);

    pool->tasks[pool->tail].task_function = task_function;
    pool->tasks[pool->tail].task_data = task_data;
    pool->tail = (pool->tail + 1) % pool->queue_size;
    pool->task_count++;

    pthread_cond_signal(&pool->task_available); // Signal using pthread_cond_signal
    pthread_mutex_unlock(&pool->queue_lock);
}

// Parallel insert function
void parallel_insert(ThreadPool *pool, Table *table, void *element, const char *filename, Table *related_table) {
    void **task_data = (void **)malloc(4 * sizeof(void *));
    if (!task_data) {
        perror("Failed to allocate memory for task data");
        return;
    }
    task_data[0] = table;
    task_data[1] = element;
    task_data[2] = (void *)filename;
    task_data[3] = related_table;

    add_task(pool, insert_task, task_data);
}

// Insert task function
void insert_task(void *data) {
    Table *table = ((Table **)data)[0];
    void *element = ((void **)data)[1];
    const char *filename = ((const char **)data)[2];
    Table *related_table = ((Table **)data)[3];
    FILE *log_file = ((FILE **)data)[4];
    int id;

    insert_into_table(table, element, &id, filename, related_table, log_file); // Pass log_file
    free(data); // Free the dynamically allocated task data
}

// Check if locks are compatible
bool is_lock_compatible(char requested_lock, char existing_lock) {
    if (requested_lock == 'S' && existing_lock == 'S') {
        return true; // Multiple shared locks are allowed
    } else if (requested_lock == 'X' && existing_lock == 'X') {
        return false; // Exclusive locks are not compatible
    } else if (requested_lock == 'S' && existing_lock == 'X') {
        return false; // Shared lock cannot be granted if an exclusive lock exists
    } else if (requested_lock == 'X' && existing_lock == 'S') {
        return false; // Exclusive lock cannot be granted if shared locks exist
    }
    return true; // No existing lock
}

// Function to print the current lock status
void print_lock_status(BufferPage *page) {
    if (page->lock_mode == 'S') {
        printf("faculty: ");
        for (int i = 0; i < page->s_lock_count; i++) {
            printf("s lock");
            if (i < page->s_lock_count - 1) {
                printf(", ");
            }
        }
        printf("\n");
    } else if (page->lock_mode == 'X') {
        printf("faculty: x lock\n");
    } else {
        printf("faculty: no lock\n");
    }
}

// Function to request a lock
bool request_lock(BufferPage *page, char lock_mode) {
    pthread_mutex_lock(&page->lock_mutex);

    // Check lock compatibility
    while (!is_lock_compatible(lock_mode, page->lock_mode)) {
        printf("Thread %ld: Waiting for lock (current lock: %c, requested: %c)\n",
               (long)pthread_self(), page->lock_mode, lock_mode);
        pthread_cond_wait(&page->lock_cond, &page->lock_mutex);
    }

    // Grant the lock
    if (lock_mode == 'S') {
        if (page->lock_mode == '\0') {
            page->lock_mode = 'S';
        }
        page->s_lock_count++;
    } else if (lock_mode == 'X') {
        page->lock_mode = 'X';
    }

    printf("Thread %ld: Lock granted (lock mode: %c)\n", (long)pthread_self(), lock_mode);
    print_lock_status(page); // Print the current lock status
    pthread_mutex_unlock(&page->lock_mutex);
    return true;
}

// Function to release a lock
void release_lock(BufferPage *page, char lock_mode) {
    pthread_mutex_lock(&page->lock_mutex);
    if (lock_mode == 'S') {
        page->s_lock_count--;
        if (page->s_lock_count == 0) {
            page->lock_mode = '\0';
        }
    } else if (lock_mode == 'X') {
        page->lock_mode = '\0';
    }
    printf("Thread %ld: Lock released\n", (long)pthread_self());
    print_lock_status(page); // Print the current lock status
    pthread_cond_broadcast(&page->lock_cond); // Notify waiting threads
    pthread_mutex_unlock(&page->lock_mutex);
}

// Thread function to test locks
void *thread_function(void *arg) {
    ThreadArg *thread_arg = (ThreadArg *)arg;
    BufferPage *page = thread_arg->page;
    char lock_mode = thread_arg->lock_mode;

    printf("Thread %ld: Requesting %c lock\n", (long)pthread_self(), lock_mode);
    request_lock(page, lock_mode);

    // Simulate work
    sleep(2);

    printf("Thread %ld: Releasing %c lock\n", (long)pthread_self(), lock_mode);
    release_lock(page, lock_mode);

    return NULL;
}

// Generate a transaction ID
int generate_transaction_id() {
    static int transaction_id_counter = 0;
    return ++transaction_id_counter;
}

// Create a buffer page
BufferPage *create_buffer_page(int page_id) {
    BufferPage *page = (BufferPage *)malloc(sizeof(BufferPage));
    page->page_id = page_id;
    page->is_dirty = false;
    page->pin_count = 0;
    page->is_locked = false;
    page->lock_mode = '\0';
    page->s_lock_count = 0;
    pthread_mutex_init(&page->lock_mutex, NULL);
    pthread_cond_init(&page->lock_cond, NULL);
    page->next = NULL;
    return page;
}

// Begin a transaction
Transaction *begin_transaction() {
    Transaction *transaction = (Transaction *)malloc(sizeof(Transaction));
    transaction->transaction_id = generate_transaction_id();
    transaction->locked_pages = NULL;
    return transaction;
}

// Commit a transaction
void commit_transaction(Transaction *transaction) {
    BufferPage *current = transaction->locked_pages;
    while (current != NULL) {
        // Determine the lock mode (S or X) for the current page
        char lock_mode = current->lock_mode; // Assuming the lock mode is stored in the BufferPage
        release_lock(current, lock_mode); // Pass the lock mode
        current = current->next;
    }
    free(transaction);
}

// Recover the database from the log
void recover_database(FILE *log_file, BufferPool *buffer_pool) {
    LogEntry entry;
    while (fread(&entry, sizeof(LogEntry), 1, log_file) == 1) {
        if (entry.operation_type == 'W') {
            // Restore the page to its old state
            BufferPage *page = find_page(buffer_pool, entry.page_id);
            if (page) {
                memcpy(page->data, entry.old_data, PAGE_SIZE);
            }
        }
    }
}

// Create a B-tree node
BTreeNode *create_btree_node(bool is_leaf) {
    BTreeNode *node = (BTreeNode *)malloc(sizeof(BTreeNode));
    if (!node) {
        perror("Failed to allocate memory for B-tree node");
        exit(EXIT_FAILURE);
    }
    node->num_keys = 0;
    node->is_leaf = is_leaf;
    for (int i = 0; i < B_TREE_ORDER; i++) {
        node->children[i] = NULL;
    }
    return node;
}

// Search for an entry in the B-tree
IndexEntry *btree_search(BTreeNode *root, int id) {
    if (!root) {
        printf("B-tree root is NULL.\n");
        return NULL;
    }

    int i = 0;
    while (i < root->num_keys && id > root->keys[i].id) {
        i++;
    }

    if (i < root->num_keys && id == root->keys[i].id) {
        return &root->keys[i]; // Key found
    }

    if (root->is_leaf) {
        return NULL; // Key not found
    }

    return btree_search(root->children[i], id); // Recursively search in the child
}

// Split a child node during insertion
void split_child(BTreeNode *parent, int index, BTreeNode *child) {
    // Create a new node to hold the right half of the child's keys
    BTreeNode *new_node = create_btree_node(child->is_leaf);
    new_node->num_keys = B_TREE_ORDER / 2 - 1;

    // Copy the right half of the child's keys to the new node
    for (int i = 0; i < new_node->num_keys; i++) {
        new_node->keys[i] = child->keys[i + B_TREE_ORDER / 2];
    }

    // If the child is not a leaf, copy the right half of its children
    if (!child->is_leaf) {
        for (int i = 0; i < B_TREE_ORDER / 2; i++) {
            new_node->children[i] = child->children[i + B_TREE_ORDER / 2];
        }
    }

    // Adjust the number of keys in the child node
    child->num_keys = B_TREE_ORDER / 2 - 1;

    // Shift the parent's children to make space for the new node
    for (int i = parent->num_keys; i > index; i--) {
        parent->children[i + 1] = parent->children[i];
    }

    // Insert the new node into the parent's children
    parent->children[index + 1] = new_node;

    // Shift the parent's keys to make space for the median key
    for (int i = parent->num_keys - 1; i >= index; i--) {
        parent->keys[i + 1] = parent->keys[i];
    }

    // Insert the median key from the child into the parent
    parent->keys[index] = child->keys[B_TREE_ORDER / 2 - 1];

    // Increment the number of keys in the parent
    parent->num_keys++;
}

// Insert a new key into a non-full node
void insert_non_full(BTreeNode *node, IndexEntry entry) {
    // Start from the rightmost key in the node
    int i = node->num_keys - 1;

    // If the node is a leaf, insert the key directly
    if (node->is_leaf) {
        // Shift keys to the right to make space for the new key
        while (i >= 0 && entry.id < node->keys[i].id) {
            node->keys[i + 1] = node->keys[i];
            i--;
        }

        // Insert the new key
        node->keys[i + 1] = entry;
        node->num_keys++;
    } else {
        // If the node is not a leaf, find the child to insert into
        while (i >= 0 && entry.id < node->keys[i].id) {
            i--;
        }

        // Check if the child is full
        if (node->children[i + 1]->num_keys == B_TREE_ORDER - 1) {
            // Split the child if it is full
            split_child(node, i + 1, node->children[i + 1]);

            // Determine which child to insert into after splitting
            if (entry.id > node->keys[i + 1].id) {
                i++;
            }
        }

        // Recursively insert into the appropriate child
        insert_non_full(node->children[i + 1], entry);
    }
}

// Insert a new key into the B-tree
void btree_insert(BTree *tree, IndexEntry entry) {
    BTreeNode *root = tree->root;
    if (root->num_keys == B_TREE_ORDER - 1) {
        BTreeNode *new_root = create_btree_node(false);
        new_root->children[0] = root;
        tree->root = new_root;
        split_child(new_root, 0, root);
        insert_non_full(new_root, entry);
    } else {
        insert_non_full(root, entry);
    }
}

// Hash function
unsigned int hash_function(const char *key) {
    unsigned int hash = 0;
    for (int i = 0; key[i] != '\0'; i++) {
        hash = (hash * 31) + key[i];
    }
    return hash % HASH_TABLE_SIZE;
}

void hash_table_insert(HashTable *table, const char *key, int index) {
    unsigned int bucket = hash_function(key) % table->size;
    pthread_mutex_lock(&table->bucket_locks[bucket]); // Lock the bucket

    HashEntry *entry = (HashEntry *)malloc(sizeof(HashEntry));
    if (!entry) {
        perror("Failed to allocate memory for hash entry");
        pthread_mutex_unlock(&table->bucket_locks[bucket]); // Unlock the bucket
        return;
    }

    strcpy(entry->key, key);
    entry->index = index;
    entry->next = table->buckets[bucket];
    table->buckets[bucket] = entry;

    pthread_mutex_unlock(&table->bucket_locks[bucket]); // Unlock the bucket
}

// Search in hash table
int hash_table_search(HashTable *table, const char *key) {
    unsigned int bucket = hash_function(key);
    printf("Searching for key '%s' in bucket %d\n", key, bucket);

    HashEntry *entry = table->buckets[bucket];
    while (entry != NULL) {
        printf("Comparing with entry key '%s'\n", entry->key);
        if (strcmp(entry->key, key) == 0) {
            printf("Key found at index %d\n", entry->index);
            return entry->index; // Key found
        }
        entry = entry->next;
    }

    printf("Key not found in hash table.\n");
    return -1; // Key not found
}

// Clear input buffer
void clear_input_buffer() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF); // Read and discard characters until newline or EOF
}

// Find a page in the buffer pool
BufferPage *find_page(BufferPool *buffer_pool, int page_id) {
    pthread_mutex_lock(&buffer_pool->lock);
    for (int i = 0; i < buffer_pool->num_pages; i++) {
        if (buffer_pool->pages[i].page_id == page_id) {
            pthread_mutex_unlock(&buffer_pool->lock);
            return &buffer_pool->pages[i];
        }
    }
    pthread_mutex_unlock(&buffer_pool->lock);
    return NULL;
}

BufferPage *find_page_to_evict(BufferPool *buffer_pool) {
    BufferPage *victim = NULL;
    time_t oldest = time(NULL);

    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        if (buffer_pool->pages[i].pin_count == 0 && buffer_pool->pages[i].last_used < oldest) {
            victim = &buffer_pool->pages[i];
            oldest = buffer_pool->pages[i].last_used;
        }
    }

    return victim;
}

void unpin_page(BufferPage *page) {
    if (page->pin_count > 0) {
        page->pin_count--;
    }
}

// Load a page from disk into the buffer pool
BufferPage *load_page(BufferPool *buffer_pool, int page_id, const char *filename) {
    BufferPage *page = find_page(buffer_pool, page_id);
    if (page) {
        page->last_used = time(NULL);
        return page;
    }

    if (buffer_pool->num_pages >= BUFFER_POOL_SIZE) {
        BufferPage *victim = find_page_to_evict(buffer_pool);
        if (!victim) {
            printf("Error: Buffer pool is full and no unpinned pages available.\n");
            return NULL;
        }
        if (victim->is_dirty) {
            write_page(buffer_pool, victim->page_id, filename);
        }
        victim->page_id = -1;
        victim->is_dirty = false;
        victim->pin_count = 0;
        page = victim;
    } else {
        page = &buffer_pool->pages[buffer_pool->num_pages++];
    }

    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file for reading");
        return NULL;
    }
    fseek(file, page_id * PAGE_SIZE, SEEK_SET);
    fread(page->data, PAGE_SIZE, 1, file);
    fclose(file);

    page->page_id = page_id;
    page->is_dirty = false;
    page->pin_count = 1;
    page->last_used = time(NULL);

    return page;
}

// Write a page back to disk if it is dirty
void write_page(BufferPool *buffer_pool, int page_id, const char *filename) {
    BufferPage *page = find_page(buffer_pool, page_id);
    if (page && page->is_dirty) {
        FILE *file = fopen(filename, "rb+");
        fseek(file, page_id * PAGE_SIZE, SEEK_SET);
        fwrite(page->data, PAGE_SIZE, 1, file);
        fclose(file);
        page->is_dirty = false;
    }
}

// Allocate memory dynamically
void *allocate_memory(MemoryManager *manager, size_t size) {
    MemoryBlock *block = (MemoryBlock *)malloc(sizeof(MemoryBlock));
    block->data = malloc(size);
    block->size = size;
    block->next = manager->head;
    manager->head = block;
    return block->data;
}

// Free hash table
void free_hash_table(HashTable *table) {
    for (int i = 0; i < table->size; i++) {
        HashEntry *entry = table->buckets[i];
        while (entry) {
            HashEntry *next = entry->next;
            free(entry);
            entry = next;
        }
    }
    free(table->buckets);
    table->buckets = NULL;
    table->size = 0;
}

// Free all allocated memory
void free_memory(MemoryManager *manager) {
    MemoryBlock *current = manager->head;
    while (current != NULL) {
        MemoryBlock *next = current->next;
        free(current->data);
        free(current);
        current = next;
    }
    manager->head = NULL;
}

// Initialize index
void initialize_index(const char *filename) {
    FILE *file = fopen(filename, "ab"); // Open in append mode (creates file if it doesn't exist)
    if (!file) {
        perror("Failed to initialize index file");
        exit(EXIT_FAILURE);
    }
    fclose(file);
}

// Initialize hash table
void initialize_hash_table(HashTable *table, int size) {
    table->buckets = (HashEntry **)calloc(size, sizeof(HashEntry *));
    table->size = size;
    table->bucket_locks = (pthread_mutex_t *)malloc(size * sizeof(pthread_mutex_t));
    for (int i = 0; i < size; i++) {
        pthread_mutex_init(&table->bucket_locks[i], NULL); // Initialize mutex for each bucket
    }
}

// Initialize index table
void initialize_index_table(IndexTable *index_table, int capacity) {
    index_table->entries = (IndexEntry *)malloc(capacity * sizeof(IndexEntry));
    index_table->capacity = capacity;
    index_table->count = 0;

    if (!index_table->entries) {
        perror("Failed to allocate memory for index table");
        exit(EXIT_FAILURE);
    }
}

// Resize index table
void resize_index_table(IndexTable *index_table, int new_capacity) {
    IndexEntry *new_entries = (IndexEntry *)realloc(index_table->entries, new_capacity * sizeof(IndexEntry));
    if (!new_entries) {
        perror("Failed to reallocate memory for index table");
        exit(EXIT_FAILURE);
    }
    index_table->entries = new_entries;
    index_table->capacity = new_capacity;
}

void initialize_buffer_pool(BufferPool *buffer_pool) {
    buffer_pool->num_pages = 0;
    pthread_mutex_init(&buffer_pool->lock, NULL);
    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        buffer_pool->pages[i].page_id = -1;
        buffer_pool->pages[i].is_dirty = false;
        buffer_pool->pages[i].pin_count = 0;
        pthread_mutex_init(&buffer_pool->pages[i].lock_mutex, NULL);
        pthread_cond_init(&buffer_pool->pages[i].lock_cond, NULL);
    }
}

// Resize hash table
void resize_hash_table(HashTable *table, int new_size) {
    // Allocate new buckets
    HashEntry **new_buckets = (HashEntry **)calloc(new_size, sizeof(HashEntry *));
    if (!new_buckets) {
        perror("Failed to allocate memory for new hash table");
        exit(EXIT_FAILURE);
    }

    // Rehash all entries
    for (int i = 0; i < table->size; i++) {
        HashEntry *entry = table->buckets[i];
        while (entry) {
            HashEntry *next = entry->next;
            unsigned int new_bucket = hash_function(entry->key) % new_size;
            entry->next = new_buckets[new_bucket];
            new_buckets[new_bucket] = entry;
            entry = next;
        }
    }

    // Free old buckets and update table
    free(table->buckets);      // Free the old buckets array
    table->buckets = new_buckets; // Assign the new buckets
    table->size = new_size;    // Update the size
}

// Resize table
void resize_table(Table *table, int new_capacity) {
    // Resize the data array
    void *new_data = realloc(table->data, table->element_size * new_capacity);
    if (!new_data) {
        perror("Failed to reallocate memory for table data");
        exit(EXIT_FAILURE);
    }
    table->data = new_data;

    // Resize the in_use array
    bool *new_in_use = (bool *)realloc(table->in_use, new_capacity * sizeof(bool));
    if (!new_in_use) {
        perror("Failed to reallocate memory for in_use array");
        exit(EXIT_FAILURE);
    }
    table->in_use = new_in_use;

    // Initialize the new slots in the in_use array to false
    for (int i = table->capacity; i < new_capacity; i++) {
        table->in_use[i] = false;
    }

    // Update the table's capacity
    table->capacity = new_capacity;
}

// Initialize the memory manager
void initialize_memory_manager(MemoryManager *manager) {
    manager->head = NULL;
}

// Save index table
void save_index_table(IndexTable *index_table, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open index file for writing");
        return;
    }

    if (fwrite(index_table->entries, sizeof(IndexEntry), index_table->count, file) != index_table->count) {
        perror("Failed to write index table to file");
    }

    fclose(file);
    printf("%s index table saved to %s.\n", index_table->count > 0 ? "Non-empty" : "Empty", filename);
}

// Initialize table
void initialize_table(Table *table, const char *name, size_t element_size, int capacity) {
    strcpy(table->name, name);
    table->element_size = element_size;
    table->capacity = capacity;
    table->count = 0;

    // Allocate memory for data and in_use arrays
    table->data = malloc(element_size * capacity);
    table->in_use = (bool *)malloc(capacity * sizeof(bool));

    // Initialize in_use array to false
    memset(table->in_use, 0, capacity * sizeof(bool));

    table->last_id = 0; // Initialize the last used ID to 0

    // Initialize B-tree and hash tables
    table->index_table.btree.root = create_btree_node(true); // Initialize B-tree root
    initialize_hash_table(&table->index_table.email_hash_table, HASH_TABLE_SIZE);
    initialize_hash_table(&table->index_table.phone_hash_table, HASH_TABLE_SIZE);

    // Initialize the buffer pool
    initialize_buffer_pool(&table->buffer_pool);

    // Initialize the table lock
    pthread_mutex_init(&table->lock, NULL);

    if (!table->data || !table->in_use) {
        perror("Failed to allocate memory for table");
        exit(EXIT_FAILURE);
    }
}

// Save hash table
void save_hash_table(HashTable *table, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open hash table file for writing");
        return;
    }

    for (int i = 0; i < table->size; i++) {
        HashEntry *entry = table->buckets[i];
        while (entry != NULL) {
            fwrite(entry, sizeof(HashEntry), 1, file);
            entry = entry->next;
        }
    }

    fclose(file);
    printf("Hash table saved to %s.\n", filename);
}

// Save table
void save_table(Table *table, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for writing");
        return;
    }

    for (int i = 0; i < table->capacity; i++) {
        if (table->in_use[i]) {
            fwrite((char *)table->data + i * table->element_size, table->element_size, 1, file);
        }
    }

    fclose(file);
}

// Load index table
void load_index_table(IndexTable *index_table, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("No existing index file found. Starting with an empty index table.\n");
        return;
    }

    // Clear the current index table
    index_table->count = 0;

    // Read entries from the file
    IndexEntry entry;
    while (fread(&entry, sizeof(IndexEntry), 1, file) == 1) {
        if (index_table->count >= index_table->capacity) {
            // Resize the index table if it is full
            int new_capacity = index_table->capacity * 2;
            resize_index_table(index_table, new_capacity);
        }

        index_table->entries[index_table->count] = entry;
        index_table->count++;
    }

    fclose(file);
    printf("%s index table loaded from %s.\n", index_table->count > 0 ? "Non-empty" : "Empty", filename);
}

// Load hash table
void load_hash_table(HashTable *table, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        return;
    }

    HashEntry entry;
    while (fread(&entry, sizeof(HashEntry), 1, file) == 1) {
        hash_table_insert(table, entry.key, entry.index);
    }

    fclose(file);
}

// Load table
void load_table(Table *table, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("No existing data file found for %s table. Starting with an empty table.\n", table->name);
        return;
    }

    // Clear the current table data
    table->count = 0;
    table->last_id = 0;
    memset(table->in_use, 0, table->capacity * sizeof(bool));

    // Read the data from the file and insert it into the table
    void *element = malloc(table->element_size);
    if (!element) {
        perror("Failed to allocate memory for element");
        fclose(file);
        return;
    }

    while (fread(element, table->element_size, 1, file) == 1) {
        // Resize the table if it is full
        if (table->count >= table->capacity) {
            int new_capacity = table->capacity * 2; // Double the capacity
            resize_table(table, new_capacity);
            printf("%s table resized to %d elements during loading.\n", table->name, new_capacity);
        }

        // Find the first unused slot
        for (int i = 0; i < table->capacity; i++) {
            if (!table->in_use[i]) {
                // Copy the element into the table
                memcpy((char *)table->data + i * table->element_size, element, table->element_size);
                table->in_use[i] = true; // Mark the slot as in use
                table->count++;

                // Update the last_id if necessary
                int *current_id = (int *)((char *)table->data + i * table->element_size);
                if (*current_id > table->last_id) {
                    table->last_id = *current_id;
                }

                break;
            }
        }
    }

    free(element);
    fclose(file);

    // Load hash tables
    char email_hash_filename[100], phone_hash_filename[100];
    snprintf(email_hash_filename, sizeof(email_hash_filename), "%s_email_hash.dat", table->name);
    snprintf(phone_hash_filename, sizeof(phone_hash_filename), "%s_phone_hash.dat", table->name);

    load_hash_table(&table->index_table.email_hash_table, email_hash_filename);
    load_hash_table(&table->index_table.phone_hash_table, phone_hash_filename);
}

// Save all tables
void save_all_tables(Table *faculty_table, Table *department_table, Table *student_table, Table *instructor_table, 
                     Table *course_table, Table *instructor_courses_table, Table *student_courses_table) {
    save_table(faculty_table, "faculty.dat");
    save_table(department_table, "departments.dat");

    save_table(student_table, "students.dat");
    save_table(instructor_table, "instructors.dat");

    save_table(course_table, "courses.dat");
    save_table(instructor_courses_table, "instructor_courses.dat");
    save_table(student_courses_table, "student_courses.dat");

    printf("All tables saved successfully.\n");
}

// Load all tables
void load_all_tables(Table *faculty_table, Table *department_table, Table *student_table, Table *instructor_table,
                     Table *course_table, Table *instructor_courses_table, Table *student_courses_table) {
    load_table(faculty_table, "faculty.dat");
    load_table(department_table, "departments.dat");

    load_table(student_table, "students.dat");
    load_table(instructor_table, "instructors.dat");

    load_table(course_table, "courses.dat");
    load_table(instructor_courses_table, "instructor_courses.dat");
    load_table(student_courses_table, "student_courses.dat");
}

// Free table resources
void free_table(Table *table) {
    free(table->data);
    free(table->in_use);
    free(table->index_table.entries); // Free the index table entries
    table->data = NULL;
    table->in_use = NULL;
    table->index_table.entries = NULL;
    table->capacity = 0;
    table->count = 0;
    table->last_id = 0;
    table->index_table.count = 0;
    table->index_table.capacity = 0;
}

// Generate SHA-256 hash
void generate_sha256(const char *input, unsigned char *digest) {
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    if (mdctx == NULL) {
        printf("Error initializing EVP_MD_CTX\n");
        exit(1);
    }

    if (EVP_DigestInit_ex(mdctx, EVP_sha256(), NULL) != 1 ||
        EVP_DigestUpdate(mdctx, input, strlen(input)) != 1 ||
        EVP_DigestFinal_ex(mdctx, digest, NULL) != 1) {
        printf("Error generating SHA-256 hash\n");
        EVP_MD_CTX_free(mdctx);
        exit(1);
    }

    EVP_MD_CTX_free(mdctx);
}

// Check if email is valid
int is_valid_email(const char *email) {
    // Find the '@' character
    char *at_sign = strchr(email, '@');
    if (!at_sign) return 0; // '@' not found

    // Ensure there is a dot after '@'
    char *dot = strchr(at_sign, '.');
    if (!dot) return 0; // '.' not found after '@'

    // Ensure the TLD has at least two characters
    if (strlen(dot) <= 2) return 0;

    return 1;
}

// Check if a record exists in the table
int is_present(Table *table, int id) {
    for (int i = 0; i < table->capacity; i++) {
        if (table->in_use[i]) {
            int *current_id = (int *)((char *)table->data + i * table->element_size);
            if (*current_id == id) {
                return 1; // Record exists
            }
        }
    }
    return 0; // Record not found
}

// Insert into index file
void insert_into_index(const char *filename, IndexEntry *entry) {
    FILE *file = fopen(filename, "ab"); // Open in append mode
    if (!file) {
        perror("Failed to open index file for writing");
        return;
    }

    // Write the index entry to the file
    fwrite(entry, sizeof(IndexEntry), 1, file);
    fclose(file);
}

// Check if email is unique
int is_email_unique(Table *table, const char *email) {
    for (int i = 0; i < table->capacity; i++) {
        if (table->in_use[i]) {
            if (strcmp(table->name, "Student") == 0) {
                Student *student = (Student *)((char *)table->data + i * table->element_size);
                if (strcmp(student->email, email) == 0) {
                    return 0; // Email already exists
                }
            } else if (strcmp(table->name, "Instructor") == 0) {
                Instructor *instructor = (Instructor *)((char *)table->data + i * table->element_size);
                if (strcmp(instructor->email, email) == 0) {
                    return 0; // Email already exists
                }
            }
        }
    }
    return 1; // Email is unique
}

// Check if phone is unique
int is_phone_unique(Table *table, const char *phone) {
    for (int i = 0; i < table->capacity; i++) {
        if (table->in_use[i]) {
            if (strcmp(table->name, "Student") == 0) {
                Student *student = (Student *)((char *)table->data + i * table->element_size);
                if (strcmp(student->phone, phone) == 0) {
                    return 0; // Phone already exists
                }
            } else if (strcmp(table->name, "Instructor") == 0) {
                Instructor *instructor = (Instructor *)((char *)table->data + i * table->element_size);
                if (strcmp(instructor->phone, phone) == 0) {
                    return 0; // Phone already exists
                }
            }
        }
    }
    return 1; // Phone is unique
}

bool insert_into_table(Table *table, void *element, int *id, const char *filename, Table *related_table, FILE *log_file) {

    // pthread_mutex_lock(&table->lock); // Lock the table
    pthread_mutex_lock(&table->lock); // Lock the table
    pthread_mutex_lock(&table->buffer_pool.lock); // Lock the buffer pool

    // Check if the table is full and resize if necessary
    if (table->count >= table->capacity) {
        int new_capacity = table->capacity * 2;
        resize_table(table, new_capacity);
    }

    // Find the first unused slot
    for (int i = 0; i < table->capacity; i++) {
        if (!table->in_use[i]) {
            // Assign an auto-incremented ID
            table->last_id++;
            *id = table->last_id;

            // Copy the element into the table
            memcpy((char *)table->data + i * table->element_size, element, table->element_size);
            table->in_use[i] = true;
            table->count++;

            // Insert into B-tree and hash tables
            IndexEntry entry = {*id, i};
            btree_insert(&table->index_table.btree, entry);

            printf("Element inserted successfully into %s table with ID: %d.\n", table->name, *id);

            pthread_mutex_unlock(&table->lock); // Unlock the table

            // Save the table to disk
            save_table(table, filename);
            
            pthread_mutex_unlock(&table->lock); // Unlock the table
            pthread_mutex_unlock(&table->buffer_pool.lock); // Unlock the buffer pool

            return true;
        }
    }

    pthread_mutex_unlock(&table->lock); // Unlock the table
    pthread_mutex_unlock(&table->buffer_pool.lock); // Unlock the buffer pool

    char new_value[100];
    snprintf(new_value, sizeof(new_value), "Inserted record with ID %d", *id);
    log_insert(log_file, table->name, *id, new_value, current_transaction->transaction_id);

    return false;
}

// Delete from index file
void delete_from_index_file(const char *filename, int id) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open index file for reading");
        return;
    }

    // Create a temporary file to store non-deleted entries
    FILE *temp_file = fopen("temp_index.dat", "wb");
    if (!temp_file) {
        perror("Failed to open temporary index file");
        fclose(file);
        return;
    }

    IndexEntry entry;
    while (fread(&entry, sizeof(IndexEntry), 1, file) == 1) {
        if (entry.id != id) {
            fwrite(&entry, sizeof(IndexEntry), 1, temp_file);
        }
    }

    fclose(file);
    fclose(temp_file);

    // Replace the original file with the temporary file
    remove(filename);
    rename("temp_index.dat", filename);
}

// Function to delete a key from a B-tree
void btree_delete(BTree *tree, IndexEntry entry) {
    BTreeNode *root = tree->root;
    if (!root) {
        printf("The tree is empty.\n");
        return;
    }

    // Call the recursive delete function
    btree_delete_recursive(root, entry.id);

    // If the root node has 0 keys, make its first child the new root if it exists
    if (root->num_keys == 0) {
        BTreeNode *temp = root;
        if (!root->is_leaf) {
            tree->root = root->children[0];
        } else {
            tree->root = NULL;
        }
        free(temp);
    }
}

// Recursive function to delete a key from a B-tree
void btree_delete_recursive(BTreeNode *node, int id) {
    int idx = find_key_index(node, id);

    // If the key is in the current node
    if (idx < node->num_keys && node->keys[idx].id == id) {
        if (node->is_leaf) {
            remove_from_leaf(node, idx);
        } else {
            remove_from_non_leaf(node, idx);
        }
    } else {
        // If the key is not in the current node and the node is a leaf, the key is not present
        if (node->is_leaf) {
            printf("The key %d is not present in the tree.\n", id);
            return;
        }

        // Flag to indicate if the key is in the last child
        bool flag = (idx == node->num_keys);

        // If the child where the key is supposed to exist has less than B_TREE_ORDER/2 keys, fill it
        if (node->children[idx]->num_keys < B_TREE_ORDER / 2) {
            fill_child(node, idx);
        }

        // If the last child has been merged, recurse on the (idx-1)th child
        if (flag && idx > node->num_keys) {
            btree_delete_recursive(node->children[idx - 1], id);
        } else {
            btree_delete_recursive(node->children[idx], id);
        }
    }
}

// Function to find the index of a key in a node
int find_key_index(BTreeNode *node, int id) {
    int idx = 0;
    while (idx < node->num_keys && node->keys[idx].id < id) {
        idx++;
    }
    return idx;
}

// Function to remove a key from a leaf node
void remove_from_leaf(BTreeNode *node, int idx) {
    for (int i = idx + 1; i < node->num_keys; i++) {
        node->keys[i - 1] = node->keys[i];
    }
    node->num_keys--;
}

// Function to remove a key from a non-leaf node
void remove_from_non_leaf(BTreeNode *node, int idx) {
    int id = node->keys[idx].id;

    // If the child that precedes the key has at least B_TREE_ORDER/2 keys, find the predecessor
    if (node->children[idx]->num_keys >= B_TREE_ORDER / 2) {
        IndexEntry pred = get_predecessor(node, idx);
        node->keys[idx] = pred;
        btree_delete_recursive(node->children[idx], pred.id);
    }
    // If the child that succeeds the key has at least B_TREE_ORDER/2 keys, find the successor
    else if (node->children[idx + 1]->num_keys >= B_TREE_ORDER / 2) {
        IndexEntry succ = get_successor(node, idx);
        node->keys[idx] = succ;
        btree_delete_recursive(node->children[idx + 1], succ.id);
    }
    // If both children have less than B_TREE_ORDER/2 keys, merge the key and the right child into the left child
    else {
        merge_children(node, idx);
        btree_delete_recursive(node->children[idx], id);
    }
}

// Function to get the predecessor of a key in a node
IndexEntry get_predecessor(BTreeNode *node, int idx) {
    BTreeNode *curr = node->children[idx];
    while (!curr->is_leaf) {
        curr = curr->children[curr->num_keys];
    }
    return curr->keys[curr->num_keys - 1];
}

// Function to get the successor of a key in a node
IndexEntry get_successor(BTreeNode *node, int idx) {
    BTreeNode *curr = node->children[idx + 1];
    while (!curr->is_leaf) {
        curr = curr->children[0];
    }
    return curr->keys[0];
}

// Function to fill a child that has less than B_TREE_ORDER/2 keys
void fill_child(BTreeNode *node, int idx) {
    // If the previous child has more than B_TREE_ORDER/2 keys, borrow a key from it
    if (idx != 0 && node->children[idx - 1]->num_keys >= B_TREE_ORDER / 2) {
        borrow_from_prev(node, idx);
    }
    // If the next child has more than B_TREE_ORDER/2 keys, borrow a key from it
    else if (idx != node->num_keys && node->children[idx + 1]->num_keys >= B_TREE_ORDER / 2) {
        borrow_from_next(node, idx);
    }
    // If both siblings have less than B_TREE_ORDER/2 keys, merge the child with one of its siblings
    else {
        if (idx != node->num_keys) {
            merge_children(node, idx);
        } else {
            merge_children(node, idx - 1);
        }
    }
}

// Function to borrow a key from the previous child
void borrow_from_prev(BTreeNode *node, int idx) {
    BTreeNode *child = node->children[idx];
    BTreeNode *sibling = node->children[idx - 1];

    // Move all keys in the child one step ahead
    for (int i = child->num_keys - 1; i >= 0; i--) {
        child->keys[i + 1] = child->keys[i];
    }

    // If the child is not a leaf, move all child pointers one step ahead
    if (!child->is_leaf) {
        for (int i = child->num_keys; i >= 0; i--) {
            child->children[i + 1] = child->children[i];
        }
    }

    // Move a key from the sibling to the child
    child->keys[0] = node->keys[idx - 1];

    // Move a child pointer from the sibling to the child if the child is not a leaf
    if (!child->is_leaf) {
        child->children[0] = sibling->children[sibling->num_keys];
    }

    // Move a key from the sibling to the parent
    node->keys[idx - 1] = sibling->keys[sibling->num_keys - 1];

    // Update the key counts
    child->num_keys++;
    sibling->num_keys--;
}

// Function to borrow a key from the next child
void borrow_from_next(BTreeNode *node, int idx) {
    BTreeNode *child = node->children[idx];
    BTreeNode *sibling = node->children[idx + 1];

    // Move a key from the parent to the child
    child->keys[child->num_keys] = node->keys[idx];

    // Move a child pointer from the sibling to the child if the child is not a leaf
    if (!child->is_leaf) {
        child->children[child->num_keys + 1] = sibling->children[0];
    }

    // Move a key from the sibling to the parent
    node->keys[idx] = sibling->keys[0];

    // Move all keys in the sibling one step back
    for (int i = 1; i < sibling->num_keys; i++) {
        sibling->keys[i - 1] = sibling->keys[i];
    }

    // If the sibling is not a leaf, move all child pointers one step back
    if (!sibling->is_leaf) {
        for (int i = 1; i <= sibling->num_keys; i++) {
            sibling->children[i - 1] = sibling->children[i];
        }
    }

    // Update the key counts
    child->num_keys++;
    sibling->num_keys--;
}

// Function to merge two children of a node
void merge_children(BTreeNode *node, int idx) {
    BTreeNode *child = node->children[idx];
    BTreeNode *sibling = node->children[idx + 1];

    // Move a key from the parent to the child
    child->keys[B_TREE_ORDER / 2 - 1] = node->keys[idx];

    // Move all keys from the sibling to the child
    for (int i = 0; i < sibling->num_keys; i++) {
        child->keys[i + B_TREE_ORDER / 2] = sibling->keys[i];
    }

    // If the child is not a leaf, move all child pointers from the sibling to the child
    if (!child->is_leaf) {
        for (int i = 0; i <= sibling->num_keys; i++) {
            child->children[i + B_TREE_ORDER / 2] = sibling->children[i];
        }
    }

    // Move all keys in the parent one step back
    for (int i = idx + 1; i < node->num_keys; i++) {
        node->keys[i - 1] = node->keys[i];
    }

    // Move all child pointers in the parent one step back
    for (int i = idx + 2; i <= node->num_keys; i++) {
        node->children[i - 1] = node->children[i];
    }

    // Update the key counts
    child->num_keys += sibling->num_keys + 1;
    node->num_keys--;

    // Free the sibling node
    free(sibling);
}

void hash_table_delete(HashTable *table, const char *key) {
    unsigned int bucket = hash_function(key) % table->size;
    pthread_mutex_lock(&table->bucket_locks[bucket]); // Lock the bucket

    HashEntry *prev = NULL;
    HashEntry *entry = table->buckets[bucket];

    while (entry != NULL) {
        if (strcmp(entry->key, key) == 0) {
            if (prev == NULL) {
                // If the entry is the first in the bucket, update the head
                table->buckets[bucket] = entry->next;
            } else {
                // Bypass the current entry
                prev->next = entry->next;
            }
            free(entry); // Free the deleted entry
            pthread_mutex_unlock(&table->bucket_locks[bucket]); // Unlock the bucket
            return;
        }
        prev = entry;
        entry = entry->next;
    }

    pthread_mutex_unlock(&table->bucket_locks[bucket]); // Unlock the bucket
}

// Delete from table
void delete_from_table(Table *table, int id, const char *filename, FILE *log_file) {
    // Lock the buffer pool and table to ensure thread safety
    pthread_mutex_lock(&table->buffer_pool.lock); // Lock the buffer pool
    pthread_mutex_lock(&table->lock); // Lock the table

    // Search for the record with the given ID
    for (int i = 0; i < table->capacity; i++) {
        if (table->in_use[i]) {
            int *current_id = (int *)((char *)table->data + i * table->element_size);
            if (*current_id == id) {
                // Mark the slot as unused
                table->in_use[i] = false;
                table->count--;

                // Remove the corresponding entry from the B-tree
                IndexEntry entry = {id, i};
                btree_delete(&table->index_table.btree, entry);

                // Remove the corresponding entry from the hash tables (if applicable)
                if (strcmp(table->name, "Student") == 0 || strcmp(table->name, "Instructor") == 0) {
                    Student *student = (Student *)((char *)table->data + i * table->element_size);
                    hash_table_delete(&table->index_table.email_hash_table, student->email);
                    hash_table_delete(&table->index_table.phone_hash_table, student->phone);
                }

                printf("Element with ID %d deleted successfully from %s table.\n", id, table->name);

                // Unlock the table and buffer pool
                pthread_mutex_unlock(&table->lock); // Unlock the table
                pthread_mutex_unlock(&table->buffer_pool.lock); // Unlock the buffer pool

                // Save the table to disk to persist the changes
                save_table(table, filename);

                // Log the delete
                char old_value[100];
                snprintf(old_value, sizeof(old_value), "Deleted record with ID %d", id);
                log_delete(log_file, table->name, id, old_value, current_transaction->transaction_id);

                return;
            }
        }
    }

    // If the record is not found, unlock the table and buffer pool
    pthread_mutex_unlock(&table->lock); // Unlock the table
    pthread_mutex_unlock(&table->buffer_pool.lock); // Unlock the buffer pool

    printf("Element with ID %d not found in %s table.\n", id, table->name);
}

// Display table from file
void display_table_from_file(const char *filename, size_t element_size, void (*print_element)(void *)) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("No existing data file found. Starting with an empty table.\n");
        return;
    }

    printf("\n***** Displaying Data from %s *****\n", filename);

    void *element = malloc(element_size);
    if (!element) {
        perror("Failed to allocate memory for element");
        fclose(file);
        return;
    }

    while (fread(element, element_size, 1, file) == 1) {
        // Check if the element is valid (e.g., ID is not 0)
        int *id = (int *)element; // Assuming the first field in the struct is the ID
        if (*id != 0) { // Skip empty or unused entries
            print_element(element);
        }
    }

    free(element);
    fclose(file);
}

// Select by ID (linear search)
void *select_by_id_linear(Table *table, int id, void (*print_element)(void *)) {
    pthread_mutex_lock(&table->lock); // Lock the table

    printf("Performing linear search for ID %d...\n", id);
    for (int i = 0; i < table->capacity; i++) {
        if (table->in_use[i]) {
            int *current_id = (int *)((char *)table->data + i * table->element_size);
            if (*current_id == id) {
                void *record = (char *)table->data + i * table->element_size;
                if (print_element) {
                    printf("\nRecord found using linear search:\n");
                    print_element(record);
                }

                pthread_mutex_unlock(&table->lock); // Unlock the table

                return record;
            }
        }
    }

    pthread_mutex_unlock(&table->lock); // Unlock the table

    printf("Record with ID %d not found.\n", id);
    return NULL; // Record not found
}

void *select_by_id_task(void *arg) {
    SelectTaskData *data = (SelectTaskData *)arg;
    for (int i = 0; i < data->table->capacity; i++) {
        if (data->table->in_use[i]) {
            int *current_id = (int *)((char *)data->table->data + i * data->table->element_size);
            if (*current_id == data->id) {
                pthread_mutex_lock(data->result_mutex);
                if (!data->result) { // Ensure only the first match is stored
                    data->result = (char *)data->table->data + i * data->table->element_size;
                }
                pthread_mutex_unlock(data->result_mutex);
                break;
            }
        }
    }
    return NULL;
}

// Select by ID (B-tree search)
void *select_by_id(Table *table, int id, void (*print_element)(void *)) {
    pthread_mutex_t result_mutex = PTHREAD_MUTEX_INITIALIZER;
    SelectTaskData data = {table, id, NULL, &result_mutex};

    int num_threads = 4; // Adjust based on your system
    pthread_t threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, select_by_id_task, &data);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    if (data.result && print_element) {
        printf("\nRecord found using parallel search:\n");
        print_element(data.result);
    } else {
        printf("Record with ID %d not found.\n", id);
    }

    return data.result;
}

// Select by email
void *select_by_email(Table *table, const char *email, void (*print_element)(void *)) {
    pthread_mutex_lock(&table->lock); // Lock the table

    // Search for the email in the hash table
    int index = hash_table_search(&table->index_table.email_hash_table, email);
    if (index == -1) {
        printf("Email '%s' not found in the %s table.\n", email, table->name);

        pthread_mutex_unlock(&table->lock); // Unlock the table

        return NULL; // Email not found
    }

    // Retrieve the record using the index
    if (index >= 0 && index < table->capacity && table->in_use[index]) {
        void *record = (char *)table->data + index * table->element_size;
        if (print_element) {
            printf("\nRecord found using email '%s':\n", email);
            print_element(record);
        }

        pthread_mutex_unlock(&table->lock); // Unlock the table

        return record;
    } else {
        printf("Invalid index %d for email '%s'. Record not found.\n", index, email);

        pthread_mutex_unlock(&table->lock); // Unlock the table

        return NULL;
    }
}

void *select_by_phone(Table *table, const char *phone, void (*print_element)(void *)) {
    pthread_mutex_lock(&table->lock); // Lock the table

    // Search for the phone in the hash table
    int index = hash_table_search(&table->index_table.phone_hash_table, phone);
    if (index == -1) {
        printf("Phone '%s' not found in the %s table.\n", phone, table->name);

        pthread_mutex_unlock(&table->lock); // Unlock the table

        return NULL; // Phone not found
    }

    // Retrieve the record using the index
    if (index >= 0 && index < table->capacity && table->in_use[index]) {
        void *record = (char *)table->data + index * table->element_size;
        if (print_element) {
            printf("\nRecord found using phone '%s':\n", phone);
            print_element(record);
        }

        pthread_mutex_unlock(&table->lock); // Unlock the table

        return record;
    } else {
        printf("Invalid index %d for phone '%s'. Record not found.\n", index, phone);

        pthread_mutex_unlock(&table->lock); // Unlock the table

        return NULL;
    }
}

// Compare field values
bool compare_field(void *record, const char *field_name, const char *field_value, const char *table_name) {
    if (strcmp(table_name, "Faculty") == 0) {
        Faculty *faculty = (Faculty *)record;
        if (strcmp(field_name, "id") == 0) {
            return faculty->id == atoi(field_value);
        } else if (strcmp(field_name, "name") == 0) {
            return strcmp(faculty->name, field_value) == 0;
        } else if (strcmp(field_name, "dean") == 0) {
            return strcmp(faculty->dean, field_value) == 0;
        }
    } else if (strcmp(table_name, "Department") == 0) {
        Department *department = (Department *)record;
        if (strcmp(field_name, "id") == 0) {
            return department->id == atoi(field_value);
        } else if (strcmp(field_name, "name") == 0) {
            return strcmp(department->name, field_value) == 0;
        } else if (strcmp(field_name, "faculty_id") == 0) {
            return department->faculty_id == atoi(field_value);
        }
    } else if (strcmp(table_name, "Student") == 0) {
        Student *student = (Student *)record;
        if (strcmp(field_name, "id") == 0) {
            return student->id == atoi(field_value);
        } else if (strcmp(field_name, "first_name") == 0) {
            return strcmp(student->first_name, field_value) == 0;
        } else if (strcmp(field_name, "last_name") == 0) {
            return strcmp(student->last_name, field_value) == 0;
        } else if (strcmp(field_name, "department_id") == 0) {
            return student->department_id == atoi(field_value);
        } else if (strcmp(field_name, "date_of_birth") == 0) {
            return strcmp(student->date_of_birth, field_value) == 0;
        } else if (strcmp(field_name, "age") == 0) {
            return strcmp(student->age, field_value) == 0;
        } else if (strcmp(field_name, "passed_hours") == 0) {
            return student->passed_hours == atof(field_value);
        } else if (strcmp(field_name, "country") == 0) {
            return strcmp(student->country, field_value) == 0;
        } else if (strcmp(field_name, "city") == 0) {
            return strcmp(student->city, field_value) == 0;
        } else if (strcmp(field_name, "street") == 0) {
            return strcmp(student->street, field_value) == 0;
        }
    } else if (strcmp(table_name, "Instructor") == 0) {
        Instructor *instructor = (Instructor *)record;
        if (strcmp(field_name, "id") == 0) {
            return instructor->id == atoi(field_value);
        } else if (strcmp(field_name, "first_name") == 0) {
            return strcmp(instructor->first_name, field_value) == 0;
        } else if (strcmp(field_name, "last_name") == 0) {
            return strcmp(instructor->last_name, field_value) == 0;
        } else if (strcmp(field_name, "department_id") == 0) {
            return instructor->department_id == atoi(field_value);
        } else if (strcmp(field_name, "date_of_birth") == 0) {
            return strcmp(instructor->date_of_birth, field_value) == 0;
        } else if (strcmp(field_name, "age") == 0) {
            return strcmp(instructor->age, field_value) == 0;
        } else if (strcmp(field_name, "salary") == 0) {
            return instructor->salary == atof(field_value);
        } else if (strcmp(field_name, "country") == 0) {
            return strcmp(instructor->country, field_value) == 0;
        } else if (strcmp(field_name, "city") == 0) {
            return strcmp(instructor->city, field_value) == 0;
        } else if (strcmp(field_name, "street") == 0) {
            return strcmp(instructor->street, field_value) == 0;
        } else if (strcmp(field_name, "SSN") == 0) {
            return strcmp(instructor->SSN, field_value) == 0;
        }
    } else if (strcmp(table_name, "Course") == 0) {
        Course *course = (Course *)record;
        if (strcmp(field_name, "id") == 0) {
            return course->id == atoi(field_value);
        } else if (strcmp(field_name, "title") == 0) {
            return strcmp(course->title, field_value) == 0;
        } else if (strcmp(field_name, "code") == 0) {
            return strcmp(course->code, field_value) == 0;
        } else if (strcmp(field_name, "active_status") == 0) {
            return strcmp(course->active_status, field_value) == 0;
        } else if (strcmp(field_name, "hours") == 0) {
            return course->hours == atof(field_value);
        } else if (strcmp(field_name, "department_id") == 0) {
            return course->department_id == atoi(field_value);
        }
    }

    // Field not found
    printf("Field '%s' not found in table '%s'.\n", field_name, table_name);
    return false;
}

// Select by field
void select_by_field(Table *table, const char *field_name, const char *field_value, void (*print_element)(void *)) {
    int found = 0;

    printf("\nSearching for records where %s = '%s':\n", field_name, field_value);

    for (int i = 0; i < table->capacity; i++) {
        if (table->in_use[i]) {
            void *record = (char *)table->data + i * table->element_size;

            // Use the generic compare_field function
            if (compare_field(record, field_name, field_value, table->name)) {
                print_element(record);
                found++;
            }
        }
    }

    if (found == 0) {
        printf("No records found where %s = '%s'.\n", field_name, field_value);
    }
}

// Update faculty
void update_faculty(Faculty *faculty) {
    int choice;
    while (1) {
        printf("\n***** Update Faculty Fields *****\n");
        printf("1. Update Name\n");
        printf("2. Update Dean\n");
        printf("3. Finish Update\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
        getchar(); // Consume leftover newline character

        switch (choice) {
            case 1: {
                printf("Enter new Faculty Name: ");
                fgets(faculty->name, sizeof(faculty->name), stdin);
                faculty->name[strcspn(faculty->name, "\n")] = 0; // Remove newline
                break;
            }
            case 2: {
                printf("Enter new Faculty Dean: ");
                fgets(faculty->dean, sizeof(faculty->dean), stdin);
                faculty->dean[strcspn(faculty->dean, "\n")] = 0; // Remove newline
                break;
            }
            case 3: {
                printf("Faculty update completed.\n");
                return; // Exit the function
            }
            default:
                printf("Invalid choice. Please try again.\n");
        }
    }
}

// Update department
void update_department(Department *department) {
    int choice;
    while (1) {
        printf("\n***** Update Department Fields *****\n");
        printf("1. Update Name\n");
        printf("2. Update Faculty ID\n");
        printf("3. Finish Update\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
        getchar(); // Consume leftover newline character

        switch (choice) {
            case 1: {
                printf("Enter new Department Name: ");
                fgets(department->name, sizeof(department->name), stdin);
                department->name[strcspn(department->name, "\n")] = 0; // Remove newline
                break;
            }
            case 2: {
                printf("Enter new Faculty ID: ");
                scanf("%d", &department->faculty_id);
                getchar(); // Consume leftover newline character
                break;
            }
            case 3: {
                printf("Department update completed.\n");
                return; // Exit the function
            }
            default:
                printf("Invalid choice. Please try again.\n");
        }
    }
}

// Update student
void update_student(Student *student) {
    int choice;
    while (1) {
        printf("\n***** Update Student Fields *****\n");
        printf("1. Update First Name\n");
        printf("2. Update Last Name\n");
        printf("3. Update Email\n");
        printf("4. Update Phone Number\n");
        printf("5. Update Date of Birth\n");
        printf("6. Update Age\n");
        printf("7. Update Passed Hours\n");
        printf("8. Update Country\n");
        printf("9. Update City\n");
        printf("10. Update Street\n");
        printf("11. Update Department ID\n");
        printf("12. Finish Update\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
        getchar(); // Consume leftover newline character

        switch (choice) {
            case 1: {
                printf("Enter new First Name: ");
                fgets(student->first_name, sizeof(student->first_name), stdin);
                student->first_name[strcspn(student->first_name, "\n")] = 0; // Remove newline
                break;
            } case 2: {
                printf("Enter new Last Name: ");
                fgets(student->last_name, sizeof(student->last_name), stdin);
                student->last_name[strcspn(student->last_name, "\n")] = 0; // Remove newline
                break;
            } case 3: {
                printf("Enter new Email: ");
                fgets(student->email, sizeof(student->email), stdin);
                student->email[strcspn(student->email, "\n")] = 0;
                break;
            } case 4: {
                printf("Enter new Phone Number: ");
                fgets(student->phone, sizeof(student->phone), stdin);
                student->phone[strcspn(student->phone, "\n")] = 0; // Remove newline
                break;
            } case 5: {
                printf("Enter new Date of Birth (YYYY-MM-DD): ");
                fgets(student->date_of_birth, sizeof(student->date_of_birth), stdin);
                student->date_of_birth[strcspn(student->date_of_birth, "\n")] = 0;
                break;
            } case 6: {
                printf("Enter new Age: ");
                fgets(student->age, sizeof(student->age), stdin);
                student->age[strcspn(student->age, "\n")] = 0;
                break;
            } case 7: {
                printf("Enter new Passed Hours: ");
                scanf("%f", &student->passed_hours);
                getchar(); // Consume the newline character
                break;
            } case 8: {
                printf("Enter new Student Country: ");
                fgets(student->country, sizeof(student->country), stdin);
                student->country[strcspn(student->country, "\n")] = 0; // Remove newline
                break;
            } case 9: {
                printf("Enter new Student City: ");
                fgets(student->city, sizeof(student->city), stdin);
                student->city[strcspn(student->city, "\n")] = 0; // Remove newline
                break;
            } case 10: {
                printf("Enter new Student Street: ");
                fgets(student->street, sizeof(student->street), stdin);
                student->street[strcspn(student->street, "\n")] = 0; // Remove newline
                break;
            } case 11: {
                printf("Enter new Department ID: ");
                scanf("%d", &student->department_id);
                getchar(); // Consume the newline character
                break;
            } case 12: {
                printf("Student update completed.\n");
                return; // Exit the function
            } default:
                printf("Invalid choice. Please try again.\n");
        }
    }
}

// Update instructor
void update_instructor(Instructor *instructor) {
    int choice;
    while (1) {
        printf("\n***** Update Instructor Fields *****\n");
        printf("1. Update First Name\n");
        printf("2. Update Last Name\n");
        printf("3. Update Email\n");
        printf("4. Update Phone Number\n");
        printf("5. Update Password\n");
        printf("6. Update Date of Birth\n");
        printf("7. Update Age\n");
        printf("8. Update SSN\n");
        printf("9. Update Salary\n");
        printf("10. Update Country\n");
        printf("11. Update City\n");
        printf("12. Update Street\n");
        printf("13. Update Department ID\n");
        printf("14. Finish Update\n");
        printf("Enter your choice: ");

        scanf("%d", &choice);
        getchar(); // Consume leftover newline character

        switch (choice) {
            case 1: {
                printf("Enter new First Name: ");
                fgets(instructor->first_name, sizeof(instructor->first_name), stdin);
                instructor->first_name[strcspn(instructor->first_name, "\n")] = 0; // Remove newline
                break;
            } case 2: {
                printf("Enter new Last Name: ");
                fgets(instructor->last_name, sizeof(instructor->last_name), stdin);
                instructor->last_name[strcspn(instructor->last_name, "\n")] = 0; // Remove newline
                break;
            } case 3: {
                printf("Enter new Email: ");
                fgets(instructor->email, sizeof(instructor->email), stdin);
                instructor->email[strcspn(instructor->email, "\n")] = 0; // Remove newline
                break;
            } case 4: {
                printf("Enter new Phone number: ");
                fgets(instructor->phone, sizeof(instructor->phone), stdin);
                instructor->phone[strcspn(instructor->phone, "\n")] = 0; // Remove newline
                break;
            } case 5: {
                char plain_password[256];
                printf("Enter the password: ");
                fgets(plain_password, sizeof(plain_password), stdin);
                plain_password[strcspn(plain_password, "\n")] = 0;

                if (strlen(plain_password) == 0) { // Check if password is empty
                    printf("Password cannot be empty.\n");
                    break;
                }

                unsigned char digest[EVP_MAX_MD_SIZE];
                generate_sha256(plain_password, digest);

                // Convert digest to a hex string for storage
                for (int i = 0; i < EVP_MD_size(EVP_sha256()); i++) {
                    sprintf(&instructor->password[i * 2], "%02x", digest[i]);
                }
                instructor->password[EVP_MD_size(EVP_sha256()) * 2] = '\0'; // Null-terminate
                break;
            } case 6: {
                printf("Enter new Date of Birth (YYYY-MM-DD): ");
                fgets(instructor->date_of_birth, sizeof(instructor->date_of_birth), stdin);
                instructor->date_of_birth[strcspn(instructor->date_of_birth, "\n")] = 0; // Remove newline
                break;
            } case 7: {
                int age;
                printf("Enter new Age: ");
                if (scanf("%d", &age) != 1 || age <= 0) {
                    printf("Invalid age. Please enter a positive integer.\n");
                    getchar();
                    break;
                }
                getchar(); // Consume leftover newline character

                // Convert integer age to string and assign
                char age_str[10];
                snprintf(age_str, sizeof(age_str), "%d", age);
                strncpy(instructor->age, age_str, sizeof(instructor->age) - 1);
                instructor->age[sizeof(instructor->age) - 1] = '\0';
                break;
            } case 8: {
                printf("Enter new SSN: ");
                fgets(instructor->SSN, sizeof(instructor->SSN), stdin);
                instructor->SSN[strcspn(instructor->SSN, "\n")] = 0; // Remove newline
                break;
            } case 9: {
                float salary;
                printf("Enter new Salary: ");
                if (scanf("%f", &salary) != 1 || salary < 0) {
                    printf("Invalid salary. Please enter a non-negative number.\n");
                    getchar(); // Consume invalid input
                    break;
                }
                instructor->salary = salary;
                getchar(); // Consume leftover newline character
                break;
            } case 10: {
                printf("Enter new Instructor Country: ");
                fgets(instructor->country, sizeof(instructor->country), stdin);
                instructor->country[strcspn(instructor->country, "\n")] = 0; // Remove newline
                break;
            } case 11: {
                printf("Enter new Instructor City: ");
                fgets(instructor->city, sizeof(instructor->city), stdin);
                instructor->city[strcspn(instructor->city, "\n")] = 0; // Remove newline
                break;
            } case 12: {
                printf("Enter new Instructor Street: ");
                fgets(instructor->street, sizeof(instructor->street), stdin);
                instructor->street[strcspn(instructor->street, "\n")] = 0; // Remove newline
                break;
            } case 13: {
                int department_id;
                printf("Enter new Department ID: ");
                if (scanf("%d", &department_id) != 1 || department_id <= 0) {
                    printf("Invalid department ID. Please enter a positive integer.\n");
                    getchar(); // Consume invalid input
                    break;
                }
                instructor->department_id = department_id;
                getchar(); // Consume leftover newline character
                break;
            } case 14: {
                printf("Instructor update completed.\n");
                return; // Exit the function
            } default:
                printf("Invalid choice. Please try again.\n");
        }
    }
}

// Update course
void update_course(Course *course) {
    int choice;
    while (1) {
        printf("\n***** Update Course Fields *****\n");
        printf("1. Update Title\n");
        printf("2. Update Course Code\n");
        printf("3. Update Active Status\n");
        printf("4. Update Hours\n");
        printf("5. Finish Update\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
        getchar(); // Consume leftover newline character

        switch (choice) {
            case 1: {
                printf("Enter new Course Title: ");
                fgets(course->title, sizeof(course->title), stdin);
                course->title[strcspn(course->title, "\n")] = 0; // Remove newline
                break;
            }
            case 2: {
                printf("Enter new Course Code: ");
                fgets(course->code, sizeof(course->code), stdin);
                course->code[strcspn(course->code, "\n")] = 0; // Remove newline
                break;
            }
            case 3: {
                printf("Enter new Active Status: ");
                fgets(course->active_status, sizeof(course->active_status), stdin);
                course->active_status[strcspn(course->active_status, "\n")] = 0; // Remove newline
                break;
            }
            case 4: {
                printf("Enter new Hours: ");
                scanf("%f", &course->hours); // Read a float value
                getchar(); // Consume leftover newline character
                break;
            }
            case 5: {
                printf("Course update completed.\n");
                return; // Exit the function
            }
            default:
                printf("Invalid choice. Please try again.\n");
        }
    }
}

// Update instructor courses
void update_instructor_courses(InstructorCourses *instructor_course, Table *course_table, Table *instructor_table) {
    int choice;
    while (1) {
        printf("\n***** Update Instructor Courses Fields *****\n");
        printf("1. Update Course ID\n");
        printf("2. Update Instructor ID\n");
        printf("3. Finish Update\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
        getchar(); // Consume leftover newline character

        switch (choice) {
            case 1: {
                int course_id;
                printf("Enter new Course ID: ");
                scanf("%d", &course_id);
                getchar(); // Consume leftover newline character

                // Check if Course ID exists in the course table
                if (!is_present(course_table, course_id)) {
                    printf("Error: Course ID %d does not exist. Update failed.\n", course_id);
                    break;
                }

                instructor_course->courses_id = course_id;
                break;
            }
            case 2: {
                int instructor_id;
                printf("Enter new Instructor ID: ");
                scanf("%d", &instructor_id);
                getchar(); // Consume leftover newline character

                // Check if Instructor ID exists in the instructor table
                if (!is_present(instructor_table, instructor_id)) {
                    printf("Error: Instructor ID %d does not exist. Update failed.\n", instructor_id);
                    break;
                }

                instructor_course->instructor_id = instructor_id;
                break;
            }
            case 3: {
                printf("Instructor Courses update completed.\n");
                return; // Exit the function
            }
            default:
                printf("Invalid choice. Please try again.\n");
        }
    }
}

// Update student courses
void update_student_courses(StudentCourses *student_course, Table *course_table, Table *student_table) {
    int choice;
    while (1) {
        printf("\n***** Update Student Courses Fields *****\n");
        printf("1. Update Course ID\n");
        printf("2. Update Student ID\n");
        printf("3. Finish Update\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
        getchar(); // Consume leftover newline character

        switch (choice) {
            case 1: {
                int course_id;
                printf("Enter new Course ID: ");
                scanf("%d", &course_id);
                getchar(); // Consume leftover newline character

                // Check if Course ID exists in the course table
                if (!is_present(course_table, course_id)) {
                    printf("Error: Course ID %d does not exist. Update failed.\n", course_id);
                    break;
                }

                student_course->courses_id = course_id;
                break;
            }
            case 2: {
                int student_id;
                printf("Enter new Student ID: ");
                scanf("%d", &student_id);
                getchar(); // Consume leftover newline character

                // Check if Student ID exists in the instructor table
                if (!is_present(student_table, student_id)) {
                    printf("Error: Student ID %d does not exist. Update failed.\n", student_id);
                    break;
                }

                student_course->student_id = student_id;
                break;
            }
            case 3: {
                printf("Student Courses update completed.\n");
                return; // Exit the function
            }
            default:
                printf("Invalid choice. Please try again.\n");
        }
    }
}

// Update record fully
void update_record_full(Table *table, int id, const char *filename, void (*print_element)(void *), 
            Table *related_table1, Table *related_table2, Table *related_table3, FILE *log_file) {
    // Lock the table and buffer pool to ensure thread safety
    pthread_mutex_lock(&table->lock); // Lock the table
    pthread_mutex_lock(&table->buffer_pool.lock); // Lock the buffer pool

    // Find the record by ID
    void *record = select_by_id(table, id, print_element);
    if (!record) {
        printf("Record with ID %d not found. Update failed.\n", id);
        
        // Unlock the table and buffer pool before returning
        pthread_mutex_unlock(&table->lock);
        pthread_mutex_unlock(&table->buffer_pool.lock);

        return;
    }

    // Display the current values of the record
    printf("\nCurrent Record:\n");
    print_element(record);

    // Update fields based on the table type
    if (strcmp(table->name, "Faculty") == 0) {
        Faculty *faculty = (Faculty *)record;
        update_faculty(faculty);
    } else if (strcmp(table->name, "Department") == 0) {
        Department *department = (Department *)record;
        update_department(department);
    } else if (strcmp(table->name, "Student") == 0) {
        Student *student = (Student *)record;
        update_student(student);
    } else if (strcmp(table->name, "Instructor") == 0) {
        Instructor *instructor = (Instructor *)record;
        update_instructor(instructor);
    } else if (strcmp(table->name, "Course") == 0) {
        Course *course = (Course *)record;
        update_course(course);
    } else if (strcmp(table->name, "InstructorCourses") == 0) {
        InstructorCourses *instructorCourse = (InstructorCourses *)record;
        if (related_table1 && related_table2) {
            update_instructor_courses(instructorCourse, related_table1, related_table2);
        } else {
            printf("Error: Course and Instructor tables are required to update Instructor Courses.\n");
        }
    } else if (strcmp(table->name, "StudentCourses") == 0) {
        StudentCourses *studentCourse = (StudentCourses *)record;
        if (related_table1 && related_table2) {
            update_student_courses(studentCourse, related_table1, related_table2);
        } else {
            printf("Error: Course and Student tables are required to update Student Courses.\n");
        }
    } else {
        printf("Update not supported for this table.\n");
        
        // Unlock the table and buffer pool before returning
        pthread_mutex_unlock(&table->lock);
        pthread_mutex_unlock(&table->buffer_pool.lock);

        return;
    }

    // Save the updated table to the file
    save_table(table, filename);
    printf("Record with ID %d updated successfully.\n", id);

    char old_value[100], new_value[100];
    snprintf(old_value, sizeof(old_value), "Old value for record %d", id);
    snprintf(new_value, sizeof(new_value), "Updated record with ID %d", id);
    log_update(log_file, table->name, id, old_value, new_value, current_transaction->transaction_id);

    // Unlock the table and buffer pool
    pthread_mutex_unlock(&table->lock);
    pthread_mutex_unlock(&table->buffer_pool.lock);
}

// Print faculty
void print_faculty(void *element) {
    Faculty *faculty = (Faculty *)element;
    printf("ID: %d, Name: %s, Dean: %s\n", faculty->id, faculty->name, faculty->dean);
}

// Print department
void print_department(void *element) {
    Department *department = (Department *)element;
    printf("ID: %d, Name: %s, Faculty ID: %d\n", department->id, department->name, department->faculty_id);
}

// Print student
void print_student(void *element) {
    Student *student = (Student *)element;
    printf("ID: %d, First Name: %s, Last Name: %s, Phone Number: %s, Email: %s, Date of Birth: %s, Passed Hours: %.2f, country: %s, city: %s, street: %s, Department ID: %d\n",
           student->id, student->first_name, student->last_name, student->phone, student->email, student->date_of_birth,
           student->passed_hours, student->country, student->city, student->street, student->department_id);
}

// Print instructor
void print_instructor(void *element) {
    Instructor *instructor = (Instructor *)element;
    printf("ID: %d, First Name: %s, Last Name: %s, Phone Number: %s, Email: %s, Date of Birth: %s, Age: %s, SSN: %s, Salary: %.2f, country: %s, city: %s, street: %s, Department ID: %d\n",
           instructor->id, instructor->first_name, instructor->last_name, instructor->phone, instructor->email, instructor->date_of_birth,
           instructor->age, instructor->SSN, instructor->salary, instructor->country, instructor->city, instructor->street, instructor->department_id);
}

// Print course
void print_course(void *element) {
    Course *course = (Course *)element;
    printf("ID: %d, Title: %s, Course Code: %s, Active Status: %s, Hours: %.2f, Department ID: %d\n",
           course->id, course->title, course->code, course->active_status, course->hours, course->department_id);
}

// Print instructor courses
void print_instructor_courses(void *element) {
    InstructorCourses *instructor_course = (InstructorCourses *)element;
    printf("ID: %d, Course ID: %d, Instructor ID: %d\n", instructor_course->id, instructor_course->courses_id, instructor_course->instructor_id);
}

// Print student courses
void print_student_courses(void *element) {
    StudentCourses *student_course = (StudentCourses *)element;
    printf("ID: %d, Course ID: %d, Student ID: %d\n", student_course->id, student_course->courses_id, student_course->student_id);
}

void input_faculty(Table *table, const char *filename, Table *related_table1, FILE *log_file) {
    Faculty faculty;

    // Input Faculty Name
    printf("Enter Faculty Name: ");
    if (fgets(faculty.name, sizeof(faculty.name), stdin) == NULL) {
        perror("Failed to read faculty name");
        return;
    }
    faculty.name[strcspn(faculty.name, "\n")] = 0; // Remove newline

    if (strlen(faculty.name) == 0) { // Check if name is empty
        printf("Name cannot be empty.\n");
        return;
    }

    // Input Faculty Dean
    printf("Enter Faculty Dean: ");
    if (fgets(faculty.dean, sizeof(faculty.dean), stdin) == NULL) {
        perror("Failed to read faculty dean");
        return;
    }
    faculty.dean[strcspn(faculty.dean, "\n")] = 0; // Remove newline

    if (strlen(faculty.dean) == 0) { // Check if dean is empty
        printf("Dean name cannot be empty.\n");
        return;
    }

    // Insert valid faculty entry into the table
    insert_into_table(table, &faculty, &faculty.id, filename, related_table1, log_file);
}

void input_department(Table *table, const char *filename, Table *related_table1, FILE *log_file) {
    Department department;

    // Input Department Name
    printf("Enter Department Name: ");
    if (fgets(department.name, sizeof(department.name), stdin) == NULL) {
        perror("Failed to read department name");
        return;
    }
    department.name[strcspn(department.name, "\n")] = 0; // Remove newline

    if (strlen(department.name) == 0) { // Check if name is empty
        printf("Department name cannot be empty.\n");
        return;
    }

    // Input Faculty ID
    char faculty_id_str[100];
    printf("Enter Faculty ID: ");
    if (fgets(faculty_id_str, sizeof(faculty_id_str), stdin) == NULL) {
        perror("Failed to read faculty ID");
        return;
    }
    faculty_id_str[strcspn(faculty_id_str, "\n")] = 0; // Remove newline

    // Convert Faculty ID to integer
    int faculty_id;
    if (sscanf(faculty_id_str, "%d", &faculty_id) != 1) {
        printf("Invalid input for Faculty ID. Please enter a valid integer.\n");
        return;
    }

    // Check if Faculty ID exists in the Faculty table
    if (!is_present(related_table1, faculty_id)) {
        printf("Error: Faculty ID %d does not exist. Department not added.\n", faculty_id);
        return;
    }

    department.faculty_id = faculty_id;

    // Insert valid department entry into the table
    insert_into_table(table, &department, &department.id, filename, related_table1, log_file);
}

void input_student(Table *table, const char *filename, Table *related_table1, FILE *log_file) {
    Student student;

    // Input First Name
    printf("Enter First Name: ");
    if (fgets(student.first_name, sizeof(student.first_name), stdin) == NULL) {
        perror("Failed to read first name");
        return;
    }
    student.first_name[strcspn(student.first_name, "\n")] = 0; // Remove newline

    if (strlen(student.first_name) == 0) { // Check if first name is empty
        printf("First name cannot be empty.\n");
        return;
    }

    // Input Last Name
    printf("Enter Last Name: ");
    if (fgets(student.last_name, sizeof(student.last_name), stdin) == NULL) {
        perror("Failed to read last name");
        return;
    }
    student.last_name[strcspn(student.last_name, "\n")] = 0; // Remove newline

    if (strlen(student.last_name) == 0) { // Check if last name is empty
        printf("Last name cannot be empty.\n");
        return;
    }

    // Input Phone Number
    printf("Enter Phone Number: ");
    if (fgets(student.phone, sizeof(student.phone), stdin) == NULL) {
        perror("Failed to read phone number");
        return;
    }
    student.phone[strcspn(student.phone, "\n")] = 0; // Remove newline

    if (strlen(student.phone) == 0) { // Check if phone is empty
        printf("Phone number cannot be empty.\n");
        return;
    }

    if (!is_phone_unique(table, student.phone)) {
        printf("Error: Phone '%s' already exists. Student not added.\n", student.phone);
        return;
    }

    // Input Email
    printf("Enter Email: ");
    if (fgets(student.email, sizeof(student.email), stdin) == NULL) {
        perror("Failed to read email");
        return;
    }
    student.email[strcspn(student.email, "\n")] = 0; // Remove newline

    if (strlen(student.email) == 0) { // Check if email is empty
        printf("Email cannot be empty.\n");
        return;
    }

    if (!is_valid_email(student.email)) {
        printf("Invalid email format. Email must end with @____.__ (e.g., @example.com).\n");
        return;
    }

    if (!is_email_unique(table, student.email)) {
        printf("Error: Email '%s' already exists. Student not added.\n", student.email);
        return;
    }

    // Input Password
    char plain_password[256];
    printf("Enter Password: ");
    if (fgets(plain_password, sizeof(plain_password), stdin) == NULL) {
        perror("Failed to read password");
        return;
    }
    plain_password[strcspn(plain_password, "\n")] = 0; // Remove newline

    if (strlen(plain_password) == 0) { // Check if password is empty
        printf("Password cannot be empty.\n");
        return;
    }

    // Hash the password
    unsigned char digest[EVP_MAX_MD_SIZE];
    generate_sha256(plain_password, digest);

    // Convert digest to a hex string for storage
    for (int i = 0; i < EVP_MD_size(EVP_sha256()); i++) {
        sprintf(&student.password[i * 2], "%02x", digest[i]);
    }

    // Input Date of Birth
    printf("Enter Date of Birth (YYYY-MM-DD): ");
    if (fgets(student.date_of_birth, sizeof(student.date_of_birth), stdin) == NULL) {
        perror("Failed to read date of birth");
        return;
    }
    student.date_of_birth[strcspn(student.date_of_birth, "\n")] = 0; // Remove newline

    // Input Age
    printf("Enter Age: ");
    if (fgets(student.age, sizeof(student.age), stdin) == NULL) {
        perror("Failed to read age");
        return;
    }
    student.age[strcspn(student.age, "\n")] = 0; // Remove newline

    // Input Passed Hours
    printf("Enter Passed Hours: ");
    if (scanf("%f", &student.passed_hours) != 1) {
        printf("Invalid input for passed hours.\n");
        clear_input_buffer();
        return;
    }
    clear_input_buffer(); // Consume the newline character

    // Input Country
    printf("Enter Country: ");
    if (fgets(student.country, sizeof(student.country), stdin) == NULL) {
        perror("Failed to read country");
        return;
    }
    student.country[strcspn(student.country, "\n")] = 0; // Remove newline

    if (strlen(student.country) == 0) { // Check if country is empty
        printf("Country cannot be empty.\n");
        return;
    }

    // Input City
    printf("Enter City: ");
    if (fgets(student.city, sizeof(student.city), stdin) == NULL) {
        perror("Failed to read city");
        return;
    }
    student.city[strcspn(student.city, "\n")] = 0; // Remove newline

    if (strlen(student.city) == 0) { // Check if city is empty
        printf("City cannot be empty.\n");
        return;
    }

    // Input Street
    printf("Enter Street: ");
    if (fgets(student.street, sizeof(student.street), stdin) == NULL) {
        perror("Failed to read street");
        return;
    }
    student.street[strcspn(student.street, "\n")] = 0; // Remove newline

    // Input Department ID
    char department_id_str[100];
    printf("Enter Department ID: ");
    if (fgets(department_id_str, sizeof(department_id_str), stdin) == NULL) {
        perror("Failed to read department ID");
        return;
    }
    department_id_str[strcspn(department_id_str, "\n")] = 0; // Remove newline

    // Convert Department ID to integer
    int department_id;
    if (sscanf(department_id_str, "%d", &department_id) != 1) {
        printf("Invalid input for Department ID. Please enter a valid integer.\n");
        return;
    }

    // Check if Department ID exists in the Department table
    if (!is_present(related_table1, department_id)) {
        printf("Error: Department ID %d does not exist. Student not added.\n", department_id);
        return;
    }

    student.department_id = department_id;

    // Insert valid student entry into the table
    insert_into_table(table, &student, &student.id, filename, related_table1, log_file);
}

void input_instructor(Table *table, const char *filename, Table *related_table1, FILE *log_file) {
    Instructor instructor;

    // Input First Name
    printf("Enter First Name: ");
    if (fgets(instructor.first_name, sizeof(instructor.first_name), stdin) == NULL) {
        perror("Failed to read first name");
        return;
    }
    instructor.first_name[strcspn(instructor.first_name, "\n")] = 0; // Remove newline

    if (strlen(instructor.first_name) == 0) { // Check if First name is empty
        printf("First name cannot be empty.\n");
        return;
    }

    // Input Last Name
    printf("Enter Last Name: ");
    if (fgets(instructor.last_name, sizeof(instructor.last_name), stdin) == NULL) {
        perror("Failed to read last name");
        return;
    }
    instructor.last_name[strcspn(instructor.last_name, "\n")] = 0; // Remove newline

    if (strlen(instructor.last_name) == 0) { // Check if Last name is empty
        printf("Last name cannot be empty.\n");
        return;
    }

    // Input Phone Number
    printf("Enter Phone Number: ");
    if (fgets(instructor.phone, sizeof(instructor.phone), stdin) == NULL) {
        perror("Failed to read phone number");
        return;
    }
    instructor.phone[strcspn(instructor.phone, "\n")] = 0; // Remove newline

    if (strlen(instructor.phone) == 0) { // Check if phone number is empty
        printf("Phone number cannot be empty.\n");
        return;
    }

    if (!is_phone_unique(table, instructor.phone)) { // Check if phone number is unique
        printf("Error: Phone '%s' already exists. Instructor not added.\n", instructor.phone);
        return;
    }

    // Input Email
    printf("Enter Email: ");
    if (fgets(instructor.email, sizeof(instructor.email), stdin) == NULL) {
        perror("Failed to read email");
        return;
    }
    instructor.email[strcspn(instructor.email, "\n")] = 0; // Remove newline

    if (strlen(instructor.email) == 0) { // Check if email empty
        printf("Email cannot be empty.\n");
        return;
    }

    if (!is_valid_email(instructor.email)) { // Check if email is valid
        printf("Invalid email format. Email must end with @____.__ (e.g., @example.com).\n");
        return;
    }

    if (!is_email_unique(table, instructor.email)) { // Check if email is unique
        printf("Error: Email '%s' already exists. Instructor not added.\n", instructor.email);
        return;
    }

    // Input Password
    char plain_password[256];
    printf("Enter Password: ");
    if (fgets(plain_password, sizeof(plain_password), stdin) == NULL) {
        perror("Failed to read password");
        return;
    }
    plain_password[strcspn(plain_password, "\n")] = 0; // Remove newline

    if (strlen(plain_password) == 0) {  // Check if Password is empty
        printf("Password cannot be empty.\n");
        return;
    }

    // Hash the password
    unsigned char digest[EVP_MAX_MD_SIZE];
    generate_sha256(plain_password, digest);

    // Convert digest to a hex string for storage
    for (int i = 0; i < EVP_MD_size(EVP_sha256()); i++) {
        sprintf(&instructor.password[i * 2], "%02x", digest[i]);
    }

    // Input Date of Birth
    printf("Enter Date of Birth (YYYY-MM-DD): ");
    if (fgets(instructor.date_of_birth, sizeof(instructor.date_of_birth), stdin) == NULL) {
        perror("Failed to read date of birth");
        return;
    }
    instructor.date_of_birth[strcspn(instructor.date_of_birth, "\n")] = 0; // Remove newline

    // Input Age
    printf("Enter Age: ");
    if (fgets(instructor.age, sizeof(instructor.age), stdin) == NULL) {
        perror("Failed to read age");
        return;
    }
    instructor.age[strcspn(instructor.age, "\n")] = 0; // Remove newline

    // Input SSN
    printf("Enter SSN: ");
    if (fgets(instructor.SSN, sizeof(instructor.SSN), stdin) == NULL) {
        perror("Failed to read SSN");
        return;
    }
    instructor.SSN[strcspn(instructor.SSN, "\n")] = 0; // Remove newline

    if (strlen(instructor.SSN) == 0) {  // Check if SSN is empty
        printf("SSN cannot be empty.\n");
        return;
    }

    // Input Salary
    printf("Enter Salary: ");
    if (scanf("%f", &instructor.salary) != 1) {
        printf("Invalid input for salary.\n");
        clear_input_buffer();
        return;
    }
    clear_input_buffer(); // Consume the newline character

    if (instructor.salary <= 0.0) { // Check if salary is empty
        printf("Salary cannot be empty or less than 0.\n");
        return;
    }

    // Input Country
    printf("Enter Country: ");
    if (fgets(instructor.country, sizeof(instructor.country), stdin) == NULL) {
        perror("Failed to read country");
        return;
    }
    instructor.country[strcspn(instructor.country, "\n")] = 0; // Remove newline

    if (strlen(instructor.country) == 0) { // Check if country is empty
        printf("Country cannot be empty.\n");
        return;
    }

    // Input City
    printf("Enter City: ");
    if (fgets(instructor.city, sizeof(instructor.city), stdin) == NULL) {
        perror("Failed to read city");
        return;
    }
    instructor.city[strcspn(instructor.city, "\n")] = 0; // Remove newline

    if (strlen(instructor.city) == 0) { // Check if city is empty
        printf("City cannot be empty.\n");
        return;
    }

    // Input Street
    printf("Enter Street: ");
    if (fgets(instructor.street, sizeof(instructor.street), stdin) == NULL) {
        perror("Failed to read street");
        return;
    }
    instructor.street[strcspn(instructor.street, "\n")] = 0; // Remove newline

    // Input Department ID
    char department_id_str[100];
    printf("Enter Department ID: ");
    if (fgets(department_id_str, sizeof(department_id_str), stdin) == NULL) {
        perror("Failed to read department ID");
        return;
    }
    department_id_str[strcspn(department_id_str, "\n")] = 0; // Remove newline

    // Convert Department ID to integer
    int department_id;
    if (sscanf(department_id_str, "%d", &department_id) != 1) {
        printf("Invalid input for Department ID. Please enter a valid integer.\n");
        return;
    }

    // Check if Department ID exists in the Department table
    if (!is_present(related_table1, department_id)) {
        printf("Error: Department ID %d does not exist. Instructor not added.\n", department_id);
        return;
    }

    instructor.department_id = department_id;

    // Insert valid instructor entry into the table
    insert_into_table(table, &instructor, &instructor.id, filename, related_table1, log_file);
}

void input_course(Table *table, const char *filename, Table *related_table1, FILE *log_file) {
    Course course;

    // Input Course Title
    printf("Enter Course Title: ");
    if (fgets(course.title, sizeof(course.title), stdin) == NULL) {
        perror("Failed to read course title");
        return;
    }
    course.title[strcspn(course.title, "\n")] = 0; // Remove newline

    if (strlen(course.title) == 0) { // Check if title is empty
        printf("Course Title cannot be empty.\n");
        return;
    }

    // Input Course Code
    printf("Enter Course Code: ");
    if (fgets(course.code, sizeof(course.code), stdin) == NULL) {
        perror("Failed to read course code");
        return;
    }
    course.code[strcspn(course.code, "\n")] = 0; // Remove newline

    if (strlen(course.code) == 0) { // Check if course code is empty
        printf("Course Code cannot be empty.\n");
        return;
    }

    // Input Course Status
    printf("Enter Course Status: ");
    if (fgets(course.active_status, sizeof(course.active_status), stdin) == NULL) {
        perror("Failed to read course status");
        return;
    }
    course.active_status[strcspn(course.active_status, "\n")] = 0; // Remove newline

    if (strlen(course.active_status) == 0) { // Check if active status is empty
        printf("Course Status cannot be empty.\n");
        return;
    }

    // Input Course Hours
    printf("Enter Course Hours: ");
    if (scanf("%f", &course.hours) != 1) {
        printf("Invalid input for course hours.\n");
        clear_input_buffer();
        return;
    }
    clear_input_buffer(); // Consume the newline character

    if (course.hours <= 0) { // Check if hours is empty
        printf("Course hours cannot be empty or less than 0.\n");
        return;
    }

    // Input Department ID
    char department_id_str[100];
    printf("Enter Department ID: ");
    if (fgets(department_id_str, sizeof(department_id_str), stdin) == NULL) {
        perror("Failed to read department ID");
    return;
    }
    department_id_str[strcspn(department_id_str, "\n")] = 0; // Remove newline

    // Convert Department ID to integer
    int department_id;
    if (sscanf(department_id_str, "%d", &department_id) != 1) {
        printf("Invalid input for Department ID. Please enter a valid integer.\n");
        return;
    }

    // Check if Department ID exists in the Department table
    if (!is_present(related_table1, department_id)) {
        printf("Error: Department ID %d does not exist. Course not added.\n", department_id);
        return;
    }

    course.department_id = department_id;

    // Insert valid course entry into the table
    insert_into_table(table, &course, &course.id, filename, related_table1, log_file);
}

void input_instructor_course(Table *table, const char *filename, Table *related_table1, Table *related_table2, FILE *log_file) {
    InstructorCourses instructorCourse;

    // Input Course ID
    char course_id_str[100];
    printf("Enter Course ID: ");
    if (fgets(course_id_str, sizeof(course_id_str), stdin) == NULL) {
        perror("Failed to read course ID");
        return;
    }
    course_id_str[strcspn(course_id_str, "\n")] = 0; // Remove newline

    // Convert Course ID to integer
    int course_id;
    if (sscanf(course_id_str, "%d", &course_id) != 1) {
        printf("Invalid input for Course ID. Please enter a valid integer.\n");
        return;
    }

    // Check if Course ID exists in the Course table
    if (!is_present(related_table1, course_id)) {
        printf("Error: Course ID %d does not exist. Instructor Course not added.\n", course_id);
        return;
    }

    instructorCourse.courses_id = course_id;

    // Input Instructor ID
    char instructor_id_str[100];
    printf("Enter Instructor ID: ");
    if (fgets(instructor_id_str, sizeof(instructor_id_str), stdin) == NULL) {
        perror("Failed to read instructor ID");
        return;
    }
    instructor_id_str[strcspn(instructor_id_str, "\n")] = 0; // Remove newline

    // Convert Instructor ID to integer
    int instructor_id;
    if (sscanf(instructor_id_str, "%d", &instructor_id) != 1) {
        printf("Invalid input for Instructor ID. Please enter a valid integer.\n");
        return;
    }

    // Check if Instructor ID exists in the Instructor table
    if (!is_present(related_table2, instructor_id)) {
        printf("Error: Instructor ID %d does not exist. Instructor Course not added.\n", instructor_id);
        return;
    }

    instructorCourse.instructor_id = instructor_id;

    // Insert valid instructor course entry into the table
    insert_into_table(table, &instructorCourse, &instructorCourse.id, filename, related_table1, log_file);
}

void input_student_course(Table *table, const char *filename, Table *related_table1, Table *related_table2, FILE *log_file) {
    StudentCourses studentCourse;

    // Input Course ID
    char course_id_str[100];
    printf("Enter Course ID: ");
    if (fgets(course_id_str, sizeof(course_id_str), stdin) == NULL) {
        perror("Failed to read course ID");
        return;
    }
    course_id_str[strcspn(course_id_str, "\n")] = 0; // Remove newline

    // Convert Course ID to integer
    int course_id;
    if (sscanf(course_id_str, "%d", &course_id) != 1) {
        printf("Invalid input for Course ID. Please enter a valid integer.\n");
        return;
    }

    // Check if Course ID exists in the Course table
    if (!is_present(related_table1, course_id)) {
        printf("Error: Course ID %d does not exist. Student Course not added.\n", course_id);
        return;
    }

    studentCourse.courses_id = course_id;

    // Input Student ID
    char student_id_str[100];
    printf("Enter Student ID: ");
    if (fgets(student_id_str, sizeof(student_id_str), stdin) == NULL) {
        perror("Failed to read student ID");
        return;
    }
    student_id_str[strcspn(student_id_str, "\n")] = 0; // Remove newline

    // Convert Student ID to integer
    int student_id;
    if (sscanf(student_id_str, "%d", &student_id) != 1) {
        printf("Invalid input for Student ID. Please enter a valid integer.\n");
        return;
    }

    // Check if Student ID exists in the Student table
    if (!is_present(related_table2, student_id)) {
        printf("Error: Student ID %d does not exist. Student Course not added.\n", student_id);
        return;
    }

    studentCourse.student_id = student_id;

    // Insert valid student course entry into the table
    insert_into_table(table, &studentCourse, &studentCourse.id, filename, related_table1, log_file);
}

// Insert into table menu
void insert_into_table_menu(Table *table, const char *filename, Table *related_table1, Table *related_table2, FILE *log_file) {
    if (strcmp(table->name, "Faculty") == 0) {
        input_faculty(table, filename, related_table1, log_file);
    } else if (strcmp(table->name, "Department") == 0) {
        input_department(table, filename, related_table1, log_file);
    } else if (strcmp(table->name, "Student") == 0) {
        input_student(table, filename, related_table1, log_file);
    } else if (strcmp(table->name, "Instructor") == 0) {
        input_instructor(table, filename, related_table1, log_file);
    } else if (strcmp(table->name, "Course") == 0) {
        input_course(table, filename, related_table1, log_file);
    } else if (strcmp(table->name, "InstructorCourses") == 0) {
        input_instructor_course(table, filename, related_table1, related_table2, log_file);
    } else if (strcmp(table->name, "StudentCourses") == 0) {
        input_student_course(table, filename, related_table1, related_table2, log_file);
    } else {
        printf("Insert operation not supported for this table.\n");
    }
}

// Select by menu
void select_by_menu(Table *table, void (*print_element)(void *), const char *filename, size_t element_size) {
    int choice;

    printf("\n***** Select By Menu *****\n");
    printf("1. Select All\n");
    printf("2. Select By ID\n");

    // Show "Select by Email" and "Select by Phone" only for Student and Instructor tables
    if (strcmp(table->name, "Student") == 0 || strcmp(table->name, "Instructor") == 0) {
        printf("3. Select by Email\n");
        printf("4. Select by Phone\n");
        printf("5. Select by Field\n");
    } else {
        printf("3. Select by Field\n");
    }

    printf("Enter your choice: ");
    scanf("%d", &choice);
    getchar(); // Consume leftover newline character

    switch (choice) {
        case 1: {
            display_table_from_file(filename, element_size, print_element);
            break;
        } case 2: {
            int id;
            printf("Enter ID to select: ");
            scanf("%d", &id);
            getchar(); // Consume leftover newline character

            // Call select_by_id with the print_element function
            select_by_id(table, id, print_element);
            break;
        } case 3: {
            if (strcmp(table->name, "Student") == 0 || strcmp(table->name, "Instructor") == 0) {
                char email[100];
                printf("Enter Email to select: ");
                if (fgets(email, sizeof(email), stdin) == NULL) {
                    perror("Failed to read email");
                    return;
                }
                email[strcspn(email, "\n")] = 0; // Remove newline

                // Call select_by_email with the print_element function
                select_by_email(table, email, print_element);
            } else {
                // Handle "Select by Field" for other tables
                char field_name[100], field_value[100];
                printf("Enter Field Name: ");
                if (fgets(field_name, sizeof(field_name), stdin) == NULL) {
                    perror("Failed to read field name");
                    return;
                }
                field_name[strcspn(field_name, "\n")] = 0; // Remove newline

                printf("Enter Field Value: ");
                if (fgets(field_value, sizeof(field_value), stdin) == NULL) {
                    perror("Failed to read field value");
                    return;
                }
                field_value[strcspn(field_value, "\n")] = 0; // Remove newline

                // Call select_by_field with the print_element function
                select_by_field(table, field_name, field_value, print_element);
            }
            break;
        } case 4: {
            char phone[100];
            printf("Enter Phone to select: ");
            if (fgets(phone, sizeof(phone), stdin) == NULL) {
                perror("Failed to read phone");
                return;
            }
            phone[strcspn(phone, "\n")] = 0; // Remove newline

            // Call select_by_phone with the print_element function
            select_by_phone(table, phone, print_element);
            
            break;
        } case 5: {
            char field_name[100], field_value[100];
            printf("Enter Field Name: ");
            if (fgets(field_name, sizeof(field_name), stdin) == NULL) {
                perror("Failed to read field name");
                return;
            }
            field_name[strcspn(field_name, "\n")] = 0; // Remove newline

            printf("Enter Field Value: ");
            if (fgets(field_value, sizeof(field_value), stdin) == NULL) {
                perror("Failed to read field value");
                return;
            }
            field_value[strcspn(field_value, "\n")] = 0; // Remove newline

            // Call select_by_field with the print_element function
            select_by_field(table, field_name, field_value, print_element);
            break;
        } default:
            printf("Invalid choice. Please try again.\n");
    }
}

// CRUD operations for a selected table
void crud_menu(Table *table, void (*print_element)(void *), const char *filename, size_t element_size, 
               Table *related_table1, Table *related_table2, Table *related_table3, FILE *log_file) {
    int choice;

    while (1) {
        printf("\n***** %s Table CRUD Menu *****\n", table->name);
        printf("1. Insert\n");
        printf("2. Delete\n");
        printf("3. Select\n");
        printf("4. Update\n");
        printf("5. Back to Main Menu\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
        getchar(); // Consume leftover newline character

        switch (choice) {
            case 1: {
                insert_into_table_menu(table, filename, related_table1, related_table2, log_file);
                break;
            } case 2: {
                int id;
                printf("Enter Id to Delete: ");
                scanf("%d", &id);
                delete_from_table(table, id, filename, log_file);
                break;
            } case 3: {
                select_by_menu(table, print_element, filename, element_size);
                break;
            } case 4: {
                int id;
                printf("Enter ID to update: ");
                scanf("%d", &id);
                getchar(); // Consume leftover newline character
                
                update_record_full(table, id, filename, print_element, related_table1, related_table2, related_table3, log_file);
                break;
            } case 5:
                return; // Go back to the main menu
            default:
                printf("Invalid choice. Please try again.\n");
        }
    }
}

void test_lock_compatibility(char test_case) {
    BufferPage page = {0}; // Initialize page with no lock
    pthread_mutex_init(&page.lock_mutex, NULL);
    pthread_cond_init(&page.lock_cond, NULL);

    pthread_t thread1, thread2;
    ThreadArg arg1, arg2;

    // Set lock modes based on the test case
    switch (test_case) {
        case '1': // S + S
            arg1.lock_mode = 'S';
            arg2.lock_mode = 'S';
            break;
        case '2': // S + X
            arg1.lock_mode = 'S';
            arg2.lock_mode = 'X';
            break;
        case '3': // X + S
            arg1.lock_mode = 'X';
            arg2.lock_mode = 'S';
            break;
        case '4': // X + X
            arg1.lock_mode = 'X';
            arg2.lock_mode = 'X';
            break;
        default:
            printf("Invalid test case. Choose 1 (S+S), 2 (S+X), 3 (X+S), or 4 (X+X).\n");
            return;
    }

    // Assign the page to both arguments
    arg1.page = &page;
    arg2.page = &page;

    // Create threads
    pthread_create(&thread1, NULL, thread_function, &arg1);
    pthread_create(&thread2, NULL, thread_function, &arg2);

    // Wait for threads to finish
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    // Clean up
    pthread_mutex_destroy(&page.lock_mutex);
    pthread_cond_destroy(&page.lock_cond);
}

void print_buffer_pool_state(BufferPool *buffer_pool) {
    printf("Buffer Pool State:\n");
    for (int i = 0; i < buffer_pool->num_pages; i++) {
        printf("Page %d: ID = %d, Pin Count = %d, Dirty = %d\n",
               i, buffer_pool->pages[i].page_id,
               buffer_pool->pages[i].pin_count,
               buffer_pool->pages[i].is_dirty);
    }
}

// Main menu
void main_menu(Table *faculty_table, Table *department_table, Table *student_table, Table *instructor_table, 
               Table *course_table, Table *instructor_courses_table, Table *student_courses_table, FILE *log_file) {
    int choice;

    while (1) {
        printf("\n***** Main Menu *****\n");
        printf("1. Faculty\n");
        printf("2. Department\n");
        printf("3. Student\n");
        printf("4. Instructor\n");
        printf("5. Course\n");
        printf("6. Instructor Courses\n");
        printf("7. Student Courses\n");
        printf("8. Test Concurrency\n");
        printf("9. Exit\n");

        printf("Enter your choice: ");
        scanf("%d", &choice);
        getchar(); // Consume leftover newline character

        switch (choice) {
            case 1: { // Faculty
                crud_menu(faculty_table, print_faculty, "faculty.dat", sizeof(Faculty), NULL, NULL, NULL, log_file);
                break;
            } case 2: { // Department
                crud_menu(department_table, print_department, "departments.dat", sizeof(Department), faculty_table, NULL, NULL, log_file);
                break;
            } case 3: { // Student
                crud_menu(student_table, print_student, "students.dat", sizeof(Student), department_table, NULL, NULL, log_file);
                break;
            } case 4: {// Instructor
                crud_menu(instructor_table, print_instructor, "instructors.dat", sizeof(Instructor), department_table, NULL, NULL, log_file);
                break;
            } case 5: {// Course
                crud_menu(course_table, print_course, "courses.dat", sizeof(Course), department_table, NULL, NULL, log_file);
                break;
            } case 6: {// Instructor Courses
                crud_menu(instructor_courses_table, print_instructor_courses, "instructor_courses.dat", sizeof(InstructorCourses), course_table, instructor_table, NULL, log_file);
                break;
            } case 7: {// Student Courses
                crud_menu(student_courses_table, print_student_courses, "student_courses.dat", sizeof(StudentCourses), course_table, student_table, NULL, log_file);
                break;
            } case 8: {// Test Concurrency
                char test_case;
                printf("Choose a test case:\n");
                printf("1. S + S\n");
                printf("2. S + X\n");
                printf("3. X + S\n");
                printf("4. X + X\n");
                printf("Enter your choice (1-4): ");
                scanf(" %c", &test_case);

                test_lock_compatibility(test_case);
                break;
            } case 9: { // Exit
                printf("Exiting...\n");

                save_all_tables(faculty_table, department_table, student_table, instructor_table,
                               course_table, student_courses_table, instructor_courses_table);

                free_table(faculty_table);
                free_table(department_table);
                free_table(student_table);
                free_table(instructor_table);
                free_table(course_table);
                free_table(instructor_courses_table);
                free_table(student_courses_table);

                exit(0);

            } default:
                printf("Invalid choice. Please try again.\n");
        }
    }
}
