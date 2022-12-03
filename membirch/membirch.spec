Name: membirch
Version: 0.0.0
Release: 1
Summary: Smart pointer types for object-level copy-on-write
Vendor: Lawrence Murray <lawrence@indii.org>
License: Apache-2.0
Group: Development/Libraries/C and C++
URL: https://birch.sh
Source0: %{name}-%{version}.tar.gz

%if 0%{?suse_version}
BuildRequires: gcc-c++ autoconf automake libtool
%endif
%if 0%{?fedora} || 0%{?rhel_version} || 0%{?centos_version}
BuildRequires: gcc-c++ autoconf automake libtool
%endif
%if 0%{?mageia}
BuildRequires: gcc-c++ libgomp-devel autoconf automake libtool
%endif

%description
C++ library of smart pointer types for object-level copy-on-write.

%package -n lib%{name}-2_0_36
Summary: Shared library for MemBirch
%description -n lib%{name}-2_0_36
Shared library for MemBirch, C++ library of smart pointer types for object-level copy-on-write.

%package devel
Summary: Development files for MemBirch
Requires: %{name} == %{version} lib%{name}-2_0_36 == %{version}
%description devel
Development files for MemBirch, C++ library of smart pointer types for object-level copy-on-write.

%package devel-static
Summary: Static libraries for MemBirch
Requires: %{name}-devel == %{version}
%description devel-static
Static libraries for MemBirch, C++ library of smart pointer types for object-level copy-on-write.

%prep
%setup -n %{name}-%{version}

%build

# tweaks to link-time optimization flags to avoid RPM lint errors on static
# library builds
%define _lto_cflags -flto -ffat-lto-objects

%if 0%{?mageia} == 7
%configure2_5x --disable-assert --enable-shared --enable-static
%else
%configure --disable-assert --enable-shared --enable-static
%endif
%make_build
strip --strip-unneeded .libs/*.so

%install
%make_install

%post -n lib%{name}-2_0_36 -p /sbin/ldconfig

%postun -n lib%{name}-2_0_36 -p /sbin/ldconfig

%files -n lib%{name}-2_0_36
%license LICENSE
%{_libdir}/lib%{name}-%{version}.so

%files devel
%license LICENSE
%{_includedir}/%{name}*
%{_libdir}/lib%{name}.so

%files devel-static
%license LICENSE
%{_libdir}/lib%{name}.a

%exclude %{_libdir}/lib%{name}.la

%changelog
* Fri Dec 2 2022 Lawrence Murray <lawrence@indii.org> 1-
Initial setup.

