class BirchSqlite < Formula
  desc "SQLite wrapper library for the Birch probabilistic programming language"
  homepage "https://birch.sh"
  url "birch-sqlite-0.0.0.tar.gz"
  version "0.0.0"
  sha256 "ee1eed431b1f636d4f121a0286fd3c230cee24c559e6b2726aef9f36003b92ba"
  license "Apache-2.0"
  depends_on "autoconf" => :build
  depends_on "automake" => :build
  depends_on "libtool" => :build
  depends_on "sqlite"
  depends_on "birch-standard" => "0.0.0"

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
