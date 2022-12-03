Name: numbirch
Version: 0.0.0
Release: 1
Summary: Numerical kernels and copy-on-write arrays
Vendor: Lawrence Murray <lawrence@indii.org>
License: Apache-2.0
Group: Development/Libraries/C and C++
URL: https://birch.sh
Source0: %{name}-%{version}.tar.gz

%if 0%{?suse_version}
BuildRequires: gcc-c++ autoconf automake libtool eigen3-devel
%endif
%if 0%{?fedora} || 0%{?rhel_version} || 0%{?centos_version}
BuildRequires: gcc-c++ autoconf automake libtool eigen3-devel
%endif
%if 0%{?mageia}
BuildRequires: gcc-c++ libgomp-devel autoconf automake libtool eigen3-devel
%endif

%description
C++ library providing numerical kernels and copy-on-write arrays.

%package -n lib%{name}-0_0_0
Summary: Shared libraries for NumBirch
%description -n lib%{name}-0_0_0
Shared libraries for NumBirch, C++ library providing numerical kernels and copy-on-write arrays.

%package devel
Summary: Development files for NumBirch
Requires: %{name} == %{version} lib%{name}-0_0_0 == %{version}
%description devel
Development files for NumBirch, C++ library providing numerical kernels and copy-on-write arrays.

%package devel-static
Summary: Static libraries for NumBirch
Requires: %{name}-devel
%description devel-static
Static libraries for NumBirch, C++ library providing numerical kernels and copy-on-write arrays.

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

%post -n lib%{name}-0_0_0 -p /sbin/ldconfig

%postun -n lib%{name}-0_0_0 -p /sbin/ldconfig

%files -n lib%{name}-0_0_0
%license LICENSE
%{_libdir}/lib%{name}-single-%{version}.so
%{_libdir}/lib%{name}-%{version}.so

%files devel
%license LICENSE
%{_includedir}/%{name}*
%{_libdir}/lib%{name}-single.so
%{_libdir}/lib%{name}.so

%files devel-static
%license LICENSE
%{_libdir}/lib%{name}-single.a
%{_libdir}/lib%{name}.a

%exclude %{_libdir}/lib%{name}-single.la
%exclude %{_libdir}/lib%{name}.la

%changelog
* Fri Dec 2 2022 Lawrence Murray <lawrence@indii.org> 1-
Initial setup.

