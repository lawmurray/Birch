class Birch < Formula
  desc "Birch is a programming language for expressing probabilistic models and performing Bayesian inference. It is used by statisticians, data scientists, and machine learning engineers. Its features include automatic differentiation, automatic marginalization, and automatic conditioning. These compose into advanced Monte Carlo inference algorithms. The Birch language transpiles to C++, with multithreaded parallelism and fast copy-on-write memory management."
  homepage "https://birch.sh"
  url "https://raw.githubusercontent.com/lawmurray/homebrew-birch/master/birch-0.0.0.tar.gz"
  version "0.0.0"
  sha256 "8eca58f9c3dc05ea171576f550cf8500a25535e21cc1b4d7bcb2010c6c057d10"
  license "Apache-2.0"
  depends_on "bison" => :build
  depends_on "flex" => :build
  depends_on "autoconf" => [:build, :recommended]
  depends_on "automake" => [:build, :recommended]
  depends_on "libtool" => [:build, :recommended]
  depends_on "binutils"
  depends_on "jemalloc"
  depends_on "libyaml"
  depends_on "birch-standard" => :recommended

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
