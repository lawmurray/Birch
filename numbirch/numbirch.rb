class Numbirch < Formula
  desc "C++ library providing numerical kernels and copy-on-write arrays"
  homepage "https://birch.sh"
  url "numbirch-0.0.0.tar.gz"
  version "0.0.0"
  sha256 "3c1a80c9cc9e42ae91dbda3b87b4ed4b9ab770b9f51bbb5b9e56c4be1c83b2a4"
  license "Apache-2.0"
  depends_on "autoconf" => :build
  depends_on "automake" => :build
  depends_on "libtool" => :build
  depends_on "libomp"
  depends_on "eigen"

  def install
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
