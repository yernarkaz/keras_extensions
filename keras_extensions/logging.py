import sys
import os
from contextlib import contextmanager

class Logger(object):
    """
    Logging helper which prints to a file and a stream (e.g. stdout) simultaneously.
    Use together with redirect_std_streams() to redirect standard streams to logger, so any print(), etc. 
    statement is logged to console and file at the same time.

    Logging to file is done with a one line buffer to allow updating lines for things like progress bars.
    """
    def __init__(self, filename='logfile.log', mode='w', stream=sys.stdout):
        assert hasattr(stream, 'write') and hasattr(stream, 'flush') # basic check for valid stream, because calling redirect_std_streams() on invalid Logger can cause trace not to be printed
        self.stream = stream
        self.file = open(filename, mode)
        self.linebuf = ''

    def __del__(self):
        if len(self.linebuf) > 0:
            self.file.write(self.linebuf)
            self.file.flush()
        self.file.close()

    def write(self, message):
        # write to stream (e.g. stdout)
        self.stream.write(message)
        self.stream.flush()

        # write to file (using line buffer to avoid writing many lines for things 
        # like progress bars that are erased and updated using '\b' and/or '\r')
        for c in message:
            if c == '\b':
                self.linebuf = self.linebuf[:-1]
            elif c == '\n':
                self.linebuf += c
                if len(self.linebuf) > 0:
                    self.file.write(self.linebuf)
                    self.file.flush()
                self.linebuf = ''
            elif c == '\r':
                self.linebuf = ''
            else:
                self.linebuf += c

    def flush(self):
        pass # already flushes each write


@contextmanager
def log_to_file(filename, mode='w', stream=sys.stdout):
    """
    Context manager that temporarily redirects stdout and stderr to 
    console (stdout) and file at the same time.

    Example:
    with log_to_file('myscript.log'):
        print('Running computation...')
        run_computation()
    """
    # keep track of original
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    # create simultaneous stream and file logger
    logger = Logger(filename, mode, stream)
    # redirect stdout and stderr
    sys.stdout = logger
    sys.stderr = logger
    try:
        # run code inside 'with' statement
        yield
    finally:
        # restore original stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

#def redirect_std_streams(logger):
#    """
#    Logging helper to redirect both stdout and stderr to the same stream-like object (in particular an instance of the Logger).
#    """
#    sys.stdout = logger
#    sys.stderr = logger
