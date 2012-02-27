#    This file is part of the Minecraft Overviewer.
#
#    Minecraft Overviewer is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or (at
#    your option) any later version.
#
#    Minecraft Overviewer is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#    Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with the Overviewer.  If not, see <http://www.gnu.org/licenses/>.

"""
This module provides a way to create named "signals" that, when
emitted, call a set of registered functions. These signals also have
the ability to be intercepted, which lets Dispatchers re-route signals
back to the main process.
"""

class Signal(object):
    """A mechanism for registering functions to be called whenever
    some specified event happens. This object is designed to work with
    Dispatcher so that functions can register to always run in the
    main Python instance."""
    
    # a global list of registered signals, indexed by name
    # this is used by JobManagers to register and relay signals
    signals = {}
    
    def __init__(self, namespace, name):
        """Creates a signal. Namespace and name should be the name of
        the class this signal is for, and the name of the signal. They
        are used to create a globally-unique name."""
        
        self.namespace = namespace
        self.name = name
        self.fullname = namespace + '.' + name
        self.interceptor = None
        self.local_functions = []
        self.functions = []
        
        # register this signal
        self.signals[self.fullname] = self
    
    def register(self, func):
        """Register a function to be called when this signal is
        emitted. Functions registered in this way will always run in
        the main Python instance."""
        self.functions.append(func)
        return func
    
    def register_local(self, func):
        """Register a function to be called when this signal is
        emitted. Functions registered in this way will always run in
        the Python instance in which they were emitted."""
        self.local_functions.append(func)
        return func
    
    def set_interceptor(self, func):
        """Sets an interceptor function. This function is called
        instead of all the non-locally registered functions if it is
        present, and should be used by JobManagers to intercept signal
        emissions."""
        self.interceptor = func
        
    def emit(self, *args, **kwargs):
        """Emits the signal with the given arguments. For convenience,
        you can also call the signal object directly.
        """
        for func in self.local_functions:
            func(*args, **kwargs)
        if self.interceptor:
            self.interceptor(*args, **kwargs)
            return
        for func in self.functions:
            func(*args, **kwargs)
    
    def emit_intercepted(self, *args, **kwargs):
        """Re-emits an intercepted signal, and finishes the work that
        would have been done during the original emission. This should
        be used by Dispatchers to re-emit signals intercepted in
        worker Python instances."""
        for func in self.functions:
            func(*args, **kwargs)
    
    # convenience
    def __call__(self, *args, **kwargs):
        self.emit(*args, **kwargs)
    
    # force pickled signals to redirect to existing signals
    def __getstate__(self):
        return self.fullname
    def __setstate__(self, fullname):
        for attr in dir(self.signals[fullname]):
            if attr.startswith('_'):
                continue
            setattr(self, attr, getattr(self.signals[fullname], attr))
