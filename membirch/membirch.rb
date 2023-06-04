class Membirch < Formula
  desc "C++ library of smart pointer types for object-level copy-on-write"
  homepage "https://birch.sh"
  url "https://download.indii.org/brew/membirch-0.0.0.tar.gz"
  version "0.0.0"
  sha256 "0dbeff83d906b31160486c9571eaa9e311f6e30ee1d637f6a2f33bb6013a03d8"
  license "Apache-2.0"
  depends_on "autoconf" => :build
  depends_on "automake" => :build
  depends_on "libtool" => :build
  depends_on "libomp"

  def install
    # see internal docs of birch driver program for explanation of OpenMP args
    system "./configure", "--disable-assert",
                          "--disable-dependency-tracking",
                          "--disable-silent-rules",
                          "--prefix=#{prefix}"
    system "make", "install"
  end

  test do
    system "true"
  end
end
