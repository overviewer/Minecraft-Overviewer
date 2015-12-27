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

import time
import logging
import progressbar
import sys
import os
import json
import rcon

class Observer(object):
    """Base class that defines the observer interface.
    """

    def __init__(self):
        self._current_value = None
        self._max_value = None
        self.start_time = None
        self.end_time = None

    def start(self, max_value):
        """Signals the start of whatever process. Must be called before update
        """
        self._set_max_value(max_value)
        self.start_time = time.time()
        self.update(0)
        return self

    def is_started(self):
        return self.start_time is not None

    def finish(self):
        """Signals the end of the processes, should be called after the
        process is done.
        """
        self.end_time = time.time()

    def is_finished(self):
        return self.end_time is not None

    def is_running(self):
        return self.is_started() and not self.is_finished()

    def add(self, amount):
        """Shortcut to update by increments instead of absolute values. Zero
        amounts are ignored.
        """
        if amount:
            self.update(self.get_current_value() + amount)

    def update(self, current_value):
        """Set the progress value. Should be between 0 and max_value. Returns
        whether this update is actually displayed.
        """
        self._current_value = current_value
        return False

    def get_percentage(self):
        """Get the current progress percentage. Assumes 100% if max_value is 0
        """
        if self.get_max_value() is 0:
            return 100.0
        else:
            return self.get_current_value() * 100.0 / self.get_max_value()

    def get_current_value(self):
        return self._current_value

    def get_max_value(self):
        return self._max_value

    def _set_max_value(self, max_value):
        self._max_value = max_value

class LoggingObserver(Observer):
    """Simple observer that just outputs status through logging.
    """
    def __init__(self):
        super(Observer, self).__init__()
        #this is an easy way to make the first update() call print a line
        self.last_update = -101

        # a fake ProgressBar, for the sake of ETA
        class FakePBar(object):
            def __init__(self):
                self.maxval = None
                self.currval = 0
                self.finished = False
                self.start_time = None
                self.seconds_elapsed = 0
            def finish(self):
                self.update(self.maxval)
            def update(self, value):
                assert 0 <= value <= self.maxval
                self.currval = value
                if self.finished:
                    return False
                if not self.start_time:
                    self.start_time = time.time()
                self.seconds_elapsed = time.time() - self.start_time

                if value == self.maxval:
                    self.finished = True

        self.fake = FakePBar();
        self.eta = progressbar.ETA()

    def start(self, max_value):
        self.fake.maxval = max_value
        super(LoggingObserver, self).start(max_value)


    def finish(self):
        self.fake.finish()
        logging.info("Rendered %d of %d.  %d%% complete.  %s", self.get_max_value(),
            self.get_max_value(), 100.0, self.eta.update(self.fake))
        super(LoggingObserver, self).finish()

    def update(self, current_value):
        super(LoggingObserver, self).update(current_value)
        self.fake.update(current_value)

        if self._need_update():
            logging.info("Rendered %d of %d.  %d%% complete.  %s",
                self.get_current_value(), self.get_max_value(),
                self.get_percentage(), self.eta.update(self.fake))
            self.last_update = current_value
            return True
        return False

    def _need_update(self):
        cur_val = self.get_current_value()
        if cur_val < 100:
            return cur_val - self.last_update > 10
        elif cur_val < 500:
            return cur_val - self.last_update > 50
        else:
            return cur_val - self.last_update > 100

default_widgets = [
    progressbar.Percentage(), ' ',
    progressbar.Bar(marker='=', left='[', right=']'), ' ',
    progressbar.CounterWidget(), ' ',
    progressbar.GenericSpeed(format='%.2ft/s'), ' ',
    progressbar.ETA(prefix='eta ')
]
class ProgressBarObserver(progressbar.ProgressBar, Observer):
    """Display progress through a progressbar.
    """

    #the progress bar is only updated in increments of this for performance
    UPDATE_INTERVAL = 25

    def __init__(self, widgets=default_widgets, term_width=None, fd=sys.stderr):
        super(ProgressBarObserver, self).__init__(widgets=widgets,
            term_width=term_width, fd=fd)
        self.last_update = 0 - (self.UPDATE_INTERVAL + 1)

    def start(self, max_value):
        self._set_max_value(max_value)
        logging.info("Rendering %d total tiles." % max_value)
        super(ProgressBarObserver, self).start()

    def is_started(self):
        return self.start_time is not None

    def finish(self):
        self._end_time = time.time()
        super(ProgressBarObserver, self).finish()
        self.fd.write('\n')
        logging.info("Rendering complete!")

    def update(self, current_value):
        # maxval is an estimate, and progressbar barfs if currval > maxval
        # so...
        current_value = min(current_value, self.maxval)
        if super(ProgressBarObserver, self).update(current_value):
            self.last_update = self.get_current_value()

    percentage = Observer.get_percentage

    def get_current_value(self):
        return self.currval

    def get_max_value(self):
        return self.maxval

    def _set_max_value(self, max_value):
        self.maxval = max_value

    def _need_update(self):
        return self.get_current_value() - self.last_update > self.UPDATE_INTERVAL

class JSObserver(Observer):
    """Display progress on index.html using JavaScript
    """

    def __init__(self, outputdir, minrefresh=5, messages=False):
        """Initialise observer
        outputdir must be set to the map output directory path
        minrefresh specifies the minimum gap between requests, in seconds [optional]
        messages is a dictionary which allows the displayed messages to be customised [optional]
        """
        self.last_update = -11
        self.last_update_time = -1 
        self._current_value = -1
        self.minrefresh = 1000*minrefresh
        self.json = dict()
        
        # function to print formatted eta
        self.format = lambda seconds: '%02ih %02im %02is' % \
            (seconds // 3600, (seconds % 3600) // 60, seconds % 60)

        if (messages == False):
            self.messages=dict(totalTiles="Rendering %d tiles", renderCompleted="Render completed in %02d:%02d:%02d", renderProgress="Rendered %d of %d tiles (%d%% ETA:%s)")
        elif (isinstance(messages, dict)):
            if ('totalTiles' in messages and 'renderCompleted' in messages and 'renderProgress' in messages):
                self.messages = messages
            else:
                raise Exception("JSObserver: messages parameter must be a dictionary with three entries: totalTiles, renderCompleted and renderProgress")
        else:
            raise Exception("JSObserver: messages parameter must be a dictionary with three entries: totalTiles, renderCompleted and renderProgress")
        if not os.path.exists(outputdir):
            raise Exception("JSObserver: Output directory specified (%s) doesn't appear to exist. This should be the same as the Overviewer output directory" % outputdir)

        self.logfile = open(os.path.join(outputdir, "progress.json"), "w+", 0)
        self.json["message"]="Render starting..."
        self.json["update"]=self.minrefresh
        self.json["messageTime"]=time.time()
        json.dump(self.json, self.logfile)
        self.logfile.flush()

    def start(self, max_value):
        self.logfile.seek(0)
        self.logfile.truncate()
        self.json["message"] = self.messages["totalTiles"] % (max_value)
        self.json["update"] = self.minrefresh
        self.json["messageTime"] = time.time()
        json.dump(self.json, self.logfile)
        self.logfile.flush()
        self.start_time=time.time()
        self._set_max_value(max_value)

    def is_started(self):
        return self.start_time is not None

    def finish(self):
        """Signals the end of the processes, should be called after the
        process is done.
        """
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.logfile.seek(0)
        self.logfile.truncate()
        hours = duration // 3600
        duration = duration % 3600
        minutes = duration // 60
        seconds = duration % 60
        self.json["message"] = self.messages["renderCompleted"] % (hours, minutes, seconds)
        self.json["update"] = 60000 # The 'renderCompleted' message will always be visible (until the next render)
        self.json["messageTime"] = time.time()
        json.dump(self.json, self.logfile)
        self.logfile.close()

    def is_finished(self):
        return self.end_time is not None

    def is_running(self):
        return self.is_started() and not self.is_finished()

    def add(self, amount):
        """Shortcut to update by increments instead of absolute values. Zero
        amounts are ignored.
        """
        if amount:
            self.update(self.get_current_value() + amount)

    def update(self, current_value):
        """Set the progress value. Should be between 0 and max_value. Returns
        whether this update is actually displayed.
        """
        self._current_value = current_value
        if self._need_update():
            refresh = max(1500*(time.time() - self.last_update_time), self.minrefresh) // 1
            self.logfile.seek(0)
            self.logfile.truncate()
            if self.get_current_value():
                duration = time.time() - self.start_time
                eta = self.format(duration * self.get_max_value() / self.get_current_value() - duration)
            else:
                eta = "?"
            self.json["message"] = self.messages["renderProgress"] % (self.get_current_value(), self.get_max_value(), self.get_percentage(), str(eta))
            self.json["update"] = refresh
            self.json["messageTime"] = time.time()
            json.dump(self.json, self.logfile)
            self.logfile.flush()
            self.last_update_time = time.time()
            self.last_update = current_value
            return True
        return False

    def get_percentage(self):
        """Get the current progress percentage. Assumes 100% if max_value is 0
        """
        if self.get_max_value() is 0:
            return 100.0
        else:
            return self.get_current_value() * 100.0 / self.get_max_value()

    def get_current_value(self):
        return self._current_value

    def get_max_value(self):
        return self._max_value

    def _set_max_value(self, max_value):
        self._max_value = max_value 

    def _need_update(self):
        cur_val = self.get_current_value()
        if cur_val < 100:
            return cur_val - self.last_update > 10
        elif cur_val < 500:
            return cur_val - self.last_update > 50
        else:
            return cur_val - self.last_update > 100

class MultiplexingObserver(Observer):
    """Combine multiple observers into one.
    """
    def __init__(self, *components):
        self.components = components
        super(MultiplexingObserver, self).__init__()

    def start(self, max_value):
        for o in self.components:
            o.start(max_value)
        super(MultiplexingObserver, self).start(max_value)

    def finish(self):
        for o in self.components:
            o.finish()
        super(MultiplexingObserver, self).finish()

    def update(self, current_value):
        for o in self.components:
            o.update(current_value)
        super(MultiplexingObserver, self).update(current_value)

class ServerAnnounceObserver(Observer):
    """Send the output to a Minecraft server via FIFO or stdin"""
    def __init__(self, target='/dev/null', pct_interval=10):
        self.pct_interval = pct_interval
        self.target_handle = open(target, 'w')
        self.last_update = 0
        super(ServerAnnounceObserver, self).__init__()

    def start(self, max_value):
        self._send_output('Starting render of %d total tiles' % max_value)
        super(ServerAnnounceObserver, self).start(max_value)

    def finish(self):
        self._send_output('Render complete!')
        super(ServerAnnounceObserver, self).finish()
        self.target_handle.close()

    def update(self, current_value):
        super(ServerAnnounceObserver, self).update(current_value)
        if self._need_update():
            self._send_output('Rendered %d of %d tiles, %d%% complete' %
                (self.get_current_value(), self.get_max_value(),
                    self.get_percentage()))
            self.last_update = current_value

    def _need_update(self):
        return self.get_percentage() - \
            (self.last_update * 100.0 / self.get_max_value()) >= self.pct_interval

    def _send_output(self, output):
        self.target_handle.write('say %s\n' % output)
        self.target_handle.flush()


# Fair amount of code duplication incoming
# Perhaps both ServerAnnounceObserver and RConObserver
# could share the percentage interval code.

class RConObserver(Observer):
    """Send the output to a Minecraft server via rcon"""

    def __init__(self, target, password, port=25575, pct_interval=10):
        self.pct_interval = pct_interval
        self.conn = rcon.RConConnection(target, port)
        self.conn.login(password)
        self.last_update = 0
        super(RConObserver, self).__init__()

    def start(self, max_value):
        self._send_output("Starting render of %d total tiles" % max_value)
        super(RConObserver, self).start(max_value)

    def finish(self):
        self._send_output("Render complete!")
        super(RConObserver, self).finish()
        self.conn.close()

    def update(self, current_value):
        super(RConObserver, self).update(current_value)
        if self._need_update():
            self._send_output('Rendered %d of %d tiles, %d%% complete' %
                (self.get_current_value(), self.get_max_value(),
                    self.get_percentage()))
            self.last_update = current_value

    def _need_update(self):
        return self.get_percentage() - \
            (self.last_update * 100.0 / self.get_max_value()) >= self.pct_interval

    def _send_output(self, output):
        self.conn.command("say", output)
        
        
