Name: birch-standard
Version: 0.0.0
Release: 1
Summary: Standard library for the Birch probabilistic programming language
Vendor: Lawrence Murray <lawrence@indii.org>
License: Apache-2.0
Group: Development/Libraries/C and C++
URL: https://birch-lang.org
Source0: birch-standard-%{version}.tar.gz
%if 0%{?suse_version}
%endif
%if 0%{?fedora} || 0%{?rhel_version} || 0%{?centos_version}
%endif

%if 0%{?suse_version} || 0%{?fedora} || 0%{?rhel_version} || 0%{?centos_version}
BuildRequires: gcc-c++ autoconf automake libtool birch == %{version} membirch-devel == %{version} numbirch-devel == %{version} libyaml-devel boost-devel
%endif
%if 0%{?mageia}
BuildRequires: gcc-c++ libgomp-devel autoconf automake libtool birch == %{version} membirch-devel == %{version} numbirch-devel == %{version} yaml-devel boost-devel
%endif

%description
Standard library of the Birch probabilistic programming language.

%package -n lib%{name}-0_0_0
Summary: Shared libraries for the Birch standard library
%description -n lib%{name}-0_0_0
Shared libraries for the Birch standard library.

%package devel
Summary: Development files for the Birch standard library
Requires: lib%{name}-0_0_0 == %{version} membirch-devel == %{version} numbirch-devel == %{version} libyaml-devel boost-devel
%description devel
Development files for the Birch standard library.

%package devel-static
Summary: Static libraries for the Birch standard library
Requires: %{name}-devel
%description devel-static
Static libraries for the Birch standard library.

%prep
%setup -q -n %{name}-%{version}

%build

# tweaks to link-time optimization flags to avoid RPM lint errors on static
# library builds
%define _lto_cflags -flto -ffat-lto-objects

%configure --disable-assert --enable-shared --enable-static
%make_build
strip --strip-unneeded .libs/*.so .libs/*.a

%install
%make_install

%post -n lib%{name}-0_0_0 -p /sbin/ldconfig

%postun -n lib%{name}-0_0_0 -p /sbin/ldconfig

%files -n lib%{name}-0_0_0
%license LICENSE
%{_libdir}/lib%{name}-single-%{version}.so
%{_libdir}/lib%{name}-%{version}.so

%files devel
%license LICENSE
%{_includedir}/%{name}.hpp
%{_libdir}/lib%{name}-single.so
%{_libdir}/lib%{name}.so

%files devel-static
%license LICENSE
%{_libdir}/lib%{name}-single.a
%{_libdir}/lib%{name}.a

%exclude %{_libdir}/lib%{name}-single.la
%exclude %{_libdir}/lib%{name}.la

%changelog
* Fri Dec 2 2022 Lawrence Murray <lawrence@indii.org> - 1:0.0.0-1
Initial setup.

