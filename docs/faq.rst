==========================
Frequently Asked Questions
==========================

**The full map doesn't display even when fully zoomed out!**
    Are you using the ``-z`` or ``--zoom`` option on your commandline or
    in settings.py? If so, try removing it, or increasing the value you set.
    It's quite likely you don't need it at all. See the documentation for the
    :option:`--zoom <-z>` option.

**You've added a few feature, but it's not showing up on my map!**
    Some new features will only show up in newly-rendered areas. Use the
    :option:`--forcerender` option to update the entire map.

**How do I use this on CentOS 5?**
    CentOS 5 comes with Python 2.4, but the Overviewer needs 2.6 or higher. See
    the special instructions at :ref:`centos`

**The background color of the map is black, and I don't like it!**
    You can change this by using the :option:`--bg-color` command line option, or
    ``bg_color`` in settings.py. See the `Options <options.html>`_ page for more
    details.

