#!/usr/bin/env python
import os
import time
from datetime import datetime, timedelta
from threading import Timer

class Cache:

    class __impl:
        """ Implementation of the singleton interface """
        def __init__(self):
            self.age = datetime.now()
            BASE_PATH = reduce (lambda l,r: l + os.path.sep + r, os.path.dirname( os.path.realpath( __file__ ) ).split( os.path.sep ) )
            self.file = os.path.join( BASE_PATH, "markers.json")
            self.markers = ""
        def getMarkers(self):
            """ Test method, return singleton id """
            # Check cache age
            if self.age and datetime.now() - self.age > timedelta (seconds = 5):
                 # load file
                 t = Timer(1, self.readMarkers)
                 t.start()
                 self.age = datetime.now()

            return self.markers

        def readMarkers(self):
            # load file
            with open(self.file, 'r') as f:
                self.markers = f.read()



    # storage for the instance reference
    __instance = None

    def __init__(self):
        """ Create singleton instance """
        # Check whether we already have an instance
        if Cache.__instance is None:
            # Create and remember instance
            Cache.__instance = Cache.__impl()

        # Store instance reference as the only member in the handle
        self.__dict__['_Cache__instance'] = Cache.__instance

    def __getattr__(self, attr):
        """ Delegate access to implementation """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__instance, attr, value)

cache = Cache()

def application(environ, start_response):
    status = '200 OK'
    response_headers = [('Content-type','application/json')]
    start_response(status, response_headers)
    #return ['Hello world!\n']
    return [cache.getMarkers()]

if __name__ == '__main__':
    from flup.server.fcgi import WSGIServer
    WSGIServer(application, bindAddress=('127.0.0.1',9950)).run()