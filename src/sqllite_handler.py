from queue import Empty
import sqlite3
from multiprocessing import Process, Queue, Event
import logging

""" 
This module handles a sql lite connection in a seperate process and inserts data into the connected database.
Only a single database can be opened at the same time. Communicates with the process via queue.
"""

logger = logging.getLogger(__name__)

## Private module data
__data_q = Queue()
__data_flag = Event()
__shutdown_flag = Event()
__p = None

def __listen(database, data_q, data_flag, shutdown_flag) -> None:
    """Main listener function. Listens on a queue for cmds to execute into the database."""
    conn = None
    try:
        conn = sqlite3.connect(database)
        logger.info(f"Created connection to '{database}'.")
        while not shutdown_flag.is_set():
            #data_flag.wait()
            logger.debug("Database listener is awake.")
            while(True):    
                # Process all arrived data
                try:
                    data = data_q.get(block=False)
                except Empty:
                    data_flag.clear()
                    break
                conn.execute(*data)
                logger.debug(f"Executed cmd: {data}")
            logger.debug("Commiting.")
            conn.commit()
    finally:
        # Clean up
        if conn is not None:
            logger.info(f"Closing connection to '{database}'...")
            conn.close()

def execute(sqlstring, args):
    """Execute a string on the database."""
    global __data_flag, __data_q
    __data_q.put((sqlstring, args))
    __data_flag.set()

def init(database: str) -> None:
    """Initialise a connection to database and insert with insert string (with ? syntax)"""
    global __p, __data_q, __data_flag, __shutdown_flag

    if __p is not None:
        quit() # Close old connection first
    __p = Process(name="SQLite", target=__listen, args=(database, __data_q, __data_flag, __shutdown_flag), daemon=True)
    __p.start()

def quit():
    """Quit and clean up the database connection"""
    global __p, __data_flag, __shutdown_flag
    if __p is not None:
        # Notify listener process to finish remaining work and then shutdown
        __shutdown_flag.set()
        __data_flag.set()
        __p.join()
    __shutdown_flag.clear()
    __data_flag.clear()

def notify_listener():
    """Manual function to notify the listener, that there might be data to process"""
    global __data_flag
    __data_flag.set()
    