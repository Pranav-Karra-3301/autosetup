class Autosetup < Formula
  desc "Rust CLI for Reproducible ML Fine-Tuning Projects"
  homepage "https://github.com/Pranav-Karra-3301/autosetup"
  version "0.1.0"
  
  if OS.mac?
    if Hardware::CPU.arm?
      url "https://github.com/Pranav-Karra-3301/autosetup/releases/download/v#{version}/autosetup-macos-aarch64"
      sha256 "PLACEHOLDER_SHA256_MACOS_ARM64"
    else
      url "https://github.com/Pranav-Karra-3301/autosetup/releases/download/v#{version}/autosetup-macos-x86_64"
      sha256 "PLACEHOLDER_SHA256_MACOS_X86_64"
    end
  elsif OS.linux?
    if Hardware::CPU.arm?
      url "https://github.com/Pranav-Karra-3301/autosetup/releases/download/v#{version}/autosetup-linux-aarch64"
      sha256 "PLACEHOLDER_SHA256_LINUX_ARM64"
    else
      url "https://github.com/Pranav-Karra-3301/autosetup/releases/download/v#{version}/autosetup-linux-x86_64"
      sha256 "PLACEHOLDER_SHA256_LINUX_X86_64"
    end
  end

  def install
    bin.install "autosetup"
  end

  test do
    assert_match "autosetup #{version}", shell_output("#{bin}/autosetup --version")
    assert_match "Initialize a new ML fine-tuning project", shell_output("#{bin}/autosetup init --help")
  end
end