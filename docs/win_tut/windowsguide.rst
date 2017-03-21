====================
Windows Newbie Guide
====================
If you're running Windows and aren't as familiar with the Windows command
prompt as the rest of the documentation assumes you are, this page is for you!

The Overviewer is a *command line* tool, which means you will need to use the start.bat to run it.

**Step 1:** Download Overviewer
    Go to the `Downloads Page <http://overviewer.org/downloads>`_ and
    download the *latest* version for your architecture, either 32 bit
    or 64 bit.
    
    *This is important. If you don't know which to choose, 32 or 64,* then you
    can find out by clicking on the start menu, *right clicking* on the
    "Computer" icon or "My Computer" icon (depending on your version of
    Windows), and then selecting "Properties." Somewhere among the information
    about your computer it should tell you if you're running a *32 bit operating
    system* or *64 bit operating system*.

    .. image:: computer_properties.png
    
    .. image:: system.png

    Once you know if your computer is 32 or 64 bit, go and download the latest
    version. We make small changes all the time, and a new version is uploaded
    to that page for every change we make. It's usually best to just get the
    latest.

Okay, you've got The Overviewer downloaded.
We're half way there!

**Step 2:** Extract the Overviewer zip you downloaded.
    This is easy. I assume you know how to unzip things. Unzip the contents to
    somewhere you can find easily. You'll need to find it in the command
    prompt. It may help to leave the window with the unzipped contents open so
    you can remind yourself where it is.

    *Keep all those files together!* They're all needed to run The Overviewer.

    .. image:: extracting.png

**Step 4:** Download the start.bat from `here <https://gist.github.com/anonymous/679e76d6e45ba4007ea0>`_ and save it where you extracted Overviewer.

    .. image:: location1.png

    
    Done that? Good. Now onto rendering the map!

**Step 5** Render a map!
    Okay, so to render a map, you have to run ``start.bat`` and it will give
    you some options to fill in.
    
    .. image:: http://i.imgur.com/SNuubyb.png

    Let's say you choose 'My World':
    
    .. image:: http://i.imgur.com/zjQgVf7.png
    
    And press enter.
    Then it asks for you to supply a output folder:
    
    .. image:: http://i.imgur.com/UTRT3FF.png
    .. image:: http://i.imgur.com/RYDIpUn.png
    So we supply our folder named MCMaps.
    Then we can decide if we want to use any render options:
    
    .. image:: http://i.imgur.com/xROeBgL.png
    For now lets say yes and use smooth-lighting, you can choose multiple by using commas.
    
    .. image:: http://i.imgur.com/flSlBaS.png
    
    If everything went according to plan, The Overviewer should now be churning
    away furiously on your world, rendering thousands of image files that
    compose a map of your world.
.. image:: http://i.imgur.com/N7Rwdot.png
    When it's done, open up the file ``index.html`` in a web browser and you
    should see your map!

I hope this has been enough to get some of you Windows noobs started on The
Overviewer. Sorry there's no easy-to-use graphical interface right now. We want
to make one, we really do, but we haven't had the time and the talent to do so
yet.

The preferred way to run The Overviewer is with a *configuration file*. Without
one, you can only do the most basic of renders. Once you're ready, head to the
:doc:`../config` page to see what else The Overviewer can do. And as always,
feel free to drop by in `IRC <http://overviewer.org/irc/>`_ if you have any
questions! We're glad to help!

Common Pitfalls
---------------

     - Wrong working directory::

            "overviewer.exe" is not recognised as an internal or external
            command, operable program, or batch file.

       This is a common mistake to make, especially for people unfamiliar
       with the command line. This happens if your current working directory
       does not contain overviewer.exe. This is likely because you've forgot
       to save the start.bat to the directory where you have unzipped
       overviewer into. Re-read Step 4 for instructions on how to do that.

     - Overviewer is on a different drive than C:\

       You may have Overviewer located on a different partition than C:,
       and for some odd reason the windows command line does not accept
       "cd D:\" as a way to switch partitions. To do this, you have to just
       type the drive letter followed by a colon::

            D:

       This should switch your current working directory to D:\


Using GitHub Gist
-----------------

Sometimes, when helping people with issues with Overviewer, we'll often
ask to see the config file you're using, or, if there was an Overviewer
error, a full copy of an error message.   Unfortunately, `IRC <http://overviewer.org/irc/>`_
is not a good way to send large amounts of text.  So we often ask users
to create a `Gist <http://gist.github.com/>`_ containing the text we want
to see.  Sites like these are also called Pastebins, and you are welcome
to use your favorite pastebin site, if you'd like.

* First, go to http://gist.github.com/

* Second, paste your text into the primary text entry area:

    .. image:: gist1.png

* Third, click the 'Create Secret Gist' button.  A secret gist means that
  only someone with the exact URL can view your gist

    .. image:: gist2.png

* Finally, send us the URL.  This will let us easily view your properly formatted Gist.

    .. image:: gist3.png
