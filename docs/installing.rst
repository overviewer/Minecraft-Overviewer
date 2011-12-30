==========
Installing
==========

This page is for installing the pre-compiled binary versions of the Overviewer.
If you want to build the Overviewer from source yourself, head to :doc:`Building
<building>`. If you have already built The Overviewer, proceed to
:doc:`running`.


Windows
=======
Running Windows and don't want to compile the Overviewer? You've come to the
right place!

1. Head to the `Downloads <https://github.com/overviewer/Minecraft-Overviewer/downloads>`_ page and download the most recent Windows download for your architecture (32 or 64 bit).

2. For 32 bit you may need to install the `VC++ 2008 <http://www.microsoft.com/downloads/en/details.aspx?FamilyID=9b2da534-3e03-4391-8a4d-074b9f2bc1bf>`_ and `VC++ 2010 <http://www.microsoft.com/downloads/en/details.aspx?familyid=a7b7a05e-6de6-4d3a-a423-37bf0912db84>`_ redistributables.

   For 64 bit, you'll want these instead: `VC++ 2008 <http://www.microsoft.com/downloads/en/details.aspx?familyid=bd2a6171-e2d6-4230-b809-9a8d7548c1b6>`_ and `VC++ 2010 <http://www.microsoft.com/download/en/details.aspx?id=14632>`_

3. That's it! Proceed with instructions on :doc:`running`.

Debian / Ubuntu
===============
We provide an APT repository with pre-built Overviewer packages for
Debian and Ubuntu users. These packages should work on any system with
a ``python2.6`` package. To do this, add the following line to your
``/etc/apt/sources.list``

::

    deb http://overviewer.org/debian ./

Then run ``apt-get update`` and ``apt-get install minecraft-overviewer`` and
you're all set! See you at the :doc:`running` page!
