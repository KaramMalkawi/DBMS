# University Database Management System

## Overview
This project is a robust University Database Management System implemented in C, designed to manage university-related data such as faculties, departments, students, instructors, courses, and their relationships. It incorporates advanced database concepts including indexing (B-tree and hash tables), buffer pool management, transaction processing, logging, and thread-safe concurrent operations using POSIX threads.

## Features
* **Data Management:** Supports CRUD (Create, Read, Update, Delete) operations for entities like Faculty, Department, Student, Instructor, Course, InstructorCourses, and StudentCourses.
* **Indexing:** Utilizes B-trees for efficient ID-based searches and hash tables for fast email and phone number lookups.
* **Buffer Pool:** Implements a buffer pool with Least Recently Used (LRU) eviction policy to manage memory efficiently.
* **Concurrency:** Ensures thread safety using mutexes and condition variables for concurrent data access, with support for shared (S) and exclusive (X) locks.
* **Transaction Management:** Supports transactions with logging for crash recovery, ensuring data consistency.
* **Security:** Passwords are hashed using SHA-256 for secure storage.
* **Data Persistence:** Stores data in files with support for loading and saving tables and their indices.
* **Input Validation:** Validates email formats, unique constraints for email and phone numbers, and referential integrity for foreign keys.

## Prerequisites
* **Compiler:** GCC or any C compiler supporting C99 or later.
* **Libraries:**
    * OpenSSL for SHA-256 hashing (`libssl-dev` on Debian/Ubuntu, `openssl-devel` on Fedora).
    * POSIX Threads (`pthreads`) for concurrency (typically included in Linux/Unix systems).
* **Operating System:** Linux/Unix (due to POSIX threads and file I/O operations). Windows may require modifications or a POSIX-compliant environment like Cygwin.

## Installation
1.  **Install Dependencies:**
    * On Ubuntu/Debian:
        ```bash
        sudo apt-get update
        sudo apt-get install build-essential libssl-dev
        ```
    * On Fedora:
        ```bash
        sudo dnf install gcc openssl-devel
        ```
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/KaramMalkawi/University-Database-Management-System.git
    cd university-database
    ```
3.  **Compile the Code:**
    ```bash
    gcc -o university_db main.c -pthread -lssl -lcrypto
    ```
4.  **Run the Program:**
    ```bash
    ./university_db
    ```

## Usage
1.  **Main Menu:** Upon running, the program presents a menu to interact with different tables (Faculty, Department, Student, Instructor, Course, InstructorCourses, StudentCourses).
2.  **CRUD Operations:** Each table supports:
    * **Insert:** Add new records with input validation.
    * **Delete:** Remove records by ID, updating indices and logging the operation.
    * **Select:** Retrieve records by ID, email, phone, or custom field searches.
    * **Update:** Modify existing records with validation for related tables.
3.  **Concurrency Testing:** Option 8 in the main menu allows testing lock compatibility (S+S, S+X, X+S, X+X).
4.  **Exit:** Saves all tables to disk and frees resources before exiting.

## File Structure
* **Source Files:**
    * `main.c`: The main program containing all structures, functions, and logic.
* **Data Files (generated at runtime):**
    * `faculty.dat`, `departments.dat`, `students.dat`, `instructors.dat`, `courses.dat`, `instructor_courses.dat`, `student_courses.dat`: Store table data.
    * `<table_name>_email_hash.dat`, `<table_name>_phone_hash.dat`: Store hash table indices for email and phone.
    * `database.log`: Transaction log for recovery.

## Key Components
### Data Structures
* **Table:** Represents a database table with dynamic data storage, in-use flags, and indexing.
* **IndexTable:** Contains a B-tree for ID searches and hash tables for email/phone lookups.
* **BufferPage:** Manages 4KB pages in the buffer pool with pin counts and lock modes.
* **LogEntry:** Records transaction details (insert, update, delete) for recovery.
* **ThreadPool:** Manages parallel tasks for operations like insertions.

### Algorithms
* **B-tree:** Used for indexing IDs with insertion, deletion, and search operations ($O(\log n)$).
* **Hash Table:** Uses chaining for collision resolution, supporting fast lookups ($O(1)$ average case).
* **LRU Eviction:** Evicts least recently used pages when the buffer pool is full.
* **Locking Mechanism:** Implements shared and exclusive locks with compatibility checks to prevent deadlocks.

### Insert a Student:
1.  Select "Student" from the main menu.
2.  Choose "Insert" and provide details (name, email, phone, etc.).
3.  The system validates email/phone uniqueness, hashes the password, and checks department ID existence.
4.  The record is inserted into the table, B-tree, and hash tables, and logged.

### Search by Email:
1.  Select "Student" and choose "Select by Email."
2.  Enter an email address.
3.  The system uses the email hash table for $O(1)$ lookup and displays the record.

### Concurrency Test:
1.  Select "Test Concurrency" and choose a test case (e.g., S+S).
2.  Two threads attempt to acquire locks, demonstrating compatibility and blocking behavior.

## Limitations
* **Single File:** All logic is in one file, which may affect maintainability for large-scale extensions.
* **No SQL Interface:** The system uses a custom menu-driven interface rather than SQL queries.
* **File-Based Storage:** Lacks advanced database features like query optimization or distributed storage.
* **Platform Dependency:** POSIX threads and OpenSSL limit portability to non-POSIX systems without modifications.

## Future Improvements
* Split code into modular files (e.g., `indexing.c`, `buffer_pool.c`) for better organization.
* Add a SQL-like query parser for more flexible data retrieval.
* Implement a client-server architecture for networked access.
* Enhance recovery mechanisms with redo/undo logging for partial transaction recovery.
* Add support for Windows via cross-platform threading libraries.

## Contributing
**Contributions are welcome! Please:**
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/your-feature`).
3.  Commit changes (`git commit -m 'Add your feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or feedback, please open an issue on the GitHub repository or contact the maintainers.
