Summary: Generates large resolution images of a Minecraft map.
Name: Minecraft-Overviewer
Version: {VERSION}
Release: 1%{?dist}
Source0: %{name}-%{version}.tar.gz
License: GNU General Public License v3
Group: Development/Libraries
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot
Vendor: Andrew Brown <brownan@gmail.com>
Url: http://overviewer.org/
Requires: python-imaging, numpy
BuildRequires: python-devel, python-imaging-devel, numpy

%description
The Minecraft Overviewer is a command-line tool for rendering high-resolution
maps of Minecraft worlds. It generates a set of static html and image files and
uses the Google Maps API to display a nice interactive map.

%prep
%setup -n %{name}

%build
env CFLAGS="$RPM_OPT_FLAGS" %{__python} setup.py build

%install
%{__python} setup.py install -O1 --root=%{buildroot}
rm -rf %{buildroot}%{_defaultdocdir}/minecraft-overviewer

%clean
rm -rf %{buildroot}

%files
%defattr(-,root,root)
%{python_sitearch}/Minecraft_Overviewer-*-*.egg-info
%{python_sitearch}/overviewer_core
%{_bindir}/overviewer.py
%doc README.rst COPYING.txt sample_config.py
