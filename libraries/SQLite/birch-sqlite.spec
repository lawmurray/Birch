Name: birch-sqlite
Version: 0.0.0
Release: 1
Summary: Birch SQLite wrapper library
Vendor: Lawrence Murray <lawrence@indii.org>
License: Apache-2.0
Group: Development/Libraries/C and C++
URL: https://birch.sh
Source0: %{name}-%{version}.tar.gz

%if 0%{?suse_version}
BuildRequires: gcc-c++ autoconf automake libtool birch-standard-devel == %{version} sqlite3-devel
%endif
%if 0%{?fedora} || 0%{?rhel_version} || 0%{?centos_version}
BuildRequires: gcc-c++ autoconf automake libtool birch-standard-devel == %{version} sqlite-devel
%endif
%if 0%{?mageia}
BuildRequires: gcc-c++ libgomp-devel autoconf automake libtool birch-standard-devel == %{version} libsqlite3-devel
%endif

%description
SQLite wrapper library for the Birch probabilistic programming language.

%package -n lib%{name}-0_0_0
Summary: Shared libraries for Birch SQLite wrapper
%description -n lib%{name}-0_0_0
Shared libraries for the Birch SQLite wrapper.

%package devel
Summary: Development files for Birch SQLite wrapper
Requires: %{name} == %{version} lib%{name}-0_0_0 == %{version} birch-standard-devel >= %{version}
%description devel
Development files for the Birch SQLite wrapper.

%package devel-static
Summary: Static libraries for Birch SQLite wrapper
Requires: %{name}-devel
%description devel-static
Static libraries for the Birch SQLite wrapper.

%if 0%{?suse_version}
%debug_package
%endif

%prep
%setup -q -n %{name}-%{version}

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

