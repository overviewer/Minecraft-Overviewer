#!/bin/bash
#    This file is part of the Minecraft Overviewer.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# This file installs the required libraries for Minecraft Overviewer.

if [ ! "$UID" = "0" ]; then
    echo "Exiting, you must be root to install libraries to the system."
    exit 1
fi

# Attempt to Figure out what package system to use and which packages
# to install.
SYSTEM=
PKGS=

if [ -f /etc/redhat-release ]; then
    # RedHat/Fedora
    echo "Detected a RedHat based operating system."
    SYSTEM=yum
    PKGS="python-imaging-devel numpy"
elif [ -f /etc/debian_version ]; then
    # Debian/Ubuntu
    echo "Detected a Debian based operating system."
    SYSTEM=apt-get
    PKGS="python-imaging python-numpy"
elif [ "`uname`" = "Darwin" ]; then
    # OS X has no native package manager. Try to find MacPorts.
    echo "Detected an OS X based operating system."
    if which port > /dev/null 2>&1; then
	SYSTEM=port
	PKGS="py-pil py-numpy"
    else
	echo "You're running OS X but no package manager was discovered."
	echo "MacPorts is available at http://www.macports.org/."
	exit 1
    fi
else
    echo "Couldn't detect your operating system."
    echo "Exiting, not sure how to proceed."
    exit 1
fi

echo "Using the ${SYSTEM} package manager to install: ${PKGS}."

${SYSTEM} install ${PKGS}
