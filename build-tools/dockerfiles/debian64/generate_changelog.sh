NOW=`date '+%a, %d %b %Y %H:%M:%S +0000'`
VER=`python2 setup.py --version`
DESC=`git describe --tags`
echo "minecraft-overviewer (${VER}-0~overviewer1) unstable; urgency=low"
echo ""
echo "  * Automatically generated from Git: ${DESC}"
echo ""
echo " -- Aaron Griffith <agrif@overviewer.org>  ${NOW}"
