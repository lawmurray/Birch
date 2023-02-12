class BirchStandard < Formula
  desc "Standard library of the Birch probabilistic programming language"
  homepage "https://birch.sh"
  url "birch-standard-0.0.0.tar.gz"
  version "0.0.0"
  sha256 "a70676d48ff63b0211d1bac767fd160f42ff80a5ced0e51fd0b457f6f7aa8c33"
  license "Apache-2.0"
  depends_on "boost"
  depends_on "libyaml"
  depends_on "membirch" => "0.0.0"
  depends_on "numbirch" => "0.0.0"

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
