FROM centos:6
RUN yum -y install epel-release && yum -y install git wget python2 @'Development Tools' python-imaging-devel numpy python-sphinx mock
RUN useradd --create-home --groups mock overviewer
USER overviewer

