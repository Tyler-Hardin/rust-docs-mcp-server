{
  description = "Rust documentation MCP server";
  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    dev-environments.url = "github:Govcraft/dev-environments";
    crane = {
      url = "github:ipetkov/crane";
    };
  };
  outputs =
    inputs@{ flake-parts, nixpkgs, crane, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        # NOTE: dev-environments' rust module is intentionally not used: it puts
        # `pkgs.rustup` in the shell and runs `rustup default stable`, which
        # resolves against the caller's $HOME/.rustup (and the network) instead
        # of the flake's pinned nixpkgs. The toolchain is provided below from
        # `pkgs` so `nix develop` is hermetic and matches `nix build`.
        inputs.dev-environments.flakeModules.golang
        inputs.dev-environments.flakeModules.node
        inputs.dev-environments.flakeModules.typst
      ];
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        {
          config,
          self',
          inputs',
          pkgs,
          system,
          ...
        }:
        let
          craneLib = inputs.crane.mkLib pkgs;

          # Common arguments shared between all builds
          commonArgs = {
            src = craneLib.cleanCargoSource ./.;
            buildInputs = with pkgs; [
              openssl
            ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
              # Additional darwin specific inputs
              pkgs.darwin.apple_sdk.frameworks.Security
              pkgs.darwin.apple_sdk.frameworks.SystemConfiguration
            ];
            nativeBuildInputs = with pkgs; [
              pkg-config
              perl
            ];
          };

          # Build *just* the cargo dependencies
          cargoArtifacts = craneLib.buildDepsOnly commonArgs;

          # Build the actual crate itself
          rustdocs-mcp-server = craneLib.buildPackage (commonArgs // {
            inherit cargoArtifacts;
          });
        in
        {
          # Add the package
          packages = {
            default = rustdocs-mcp-server;
            rustdocs-mcp-server = rustdocs-mcp-server;
          };

          # Golang Development Environment Options
          # ----------------------------------
          # enable: boolean - Enable/disable the Golang environment
          # goVersion: enum - Go toolchain version ("1.18", "1.19", "1.20", "1.21, "1.22"", "1.23") (default: "1.23")
          # withTools: list of strings - Additional Go tools to include (e.g., "golint", "gopls")
          # extraPackages: list of packages - Additional packages to include
          go-dev = {
            # enable = true;
            # goVersion = "1.23";
            # withTools = [ "gopls" "golint" ];
            # extraPackages = [ ];
          };

          # Rust Development Environment
          # ----------------------------------
          # The toolchain is supplied by the flake's pinned nixpkgs (see
          # devShells.default below) rather than rustup, so the shell does not
          # depend on the caller's ~/.rustup or on network access.

          # Node.js Development Environment Options
          # -------------------------------------
          # enable: boolean - Enable/disable the Node environment
          # nodeVersion: string - Version of Node.js to use (default: "20")
          # withTools: list of strings - Global tools to include (default: ["typescript" "yarn" "pnpm"])
          # extraPackages: list of packages - Additional packages to include
          # ide.type: enum - IDE preference ("vscode", "webstorm", "none") (default: "none")
          node-dev = {
            # Example configuration:
            # enable = true;
            # nodeVersion = "20";
            # withTools = [ "typescript" "yarn" "pnpm" ];
            # extraPackages = [ ];
            # ide.type = "none";
          };

          typst-dev = {
            # Example configuration:
            # enable = true;
            # withTools = [ "typst-fmt" "typst-lsp" ];
            # extraPackages = [ ];
            # ide.type = "none";
          };
          # Create the combined shell
          devShells.default = pkgs.mkShell {
            buildInputs = nixpkgs.lib.flatten (nixpkgs.lib.attrValues config.env-packages ++ [ ])
              ++ (with pkgs; [
                rustc
                cargo
                rustfmt
                clippy
                rust-analyzer
                llvm
                cargo-tarpaulin
                cargo-release
                cargo-machete
                pkg-config
                openssl
                openssl.dev
                perl
              ]);
            # Combine existing hooks with new exports for OpenSSL
            shellHook = ''
              ${nixpkgs.lib.concatStringsSep "\n" (nixpkgs.lib.attrValues config.env-hooks)}

              # Keep cargo state out of the caller's ~/.cargo. cargo prepends
              # $CARGO_HOME/bin to PATH when resolving subcommands, so stale or
              # foreign binaries installed there (e.g. rustup shims linked
              # against a garbage-collected glibc) shadow the tools this shell
              # provides and fail with "could not execute process".
              export CARGO_HOME="''${XDG_CACHE_HOME:-$HOME/.cache}/rustdocs-mcp-server/cargo"
              mkdir -p "$CARGO_HOME"

              # Export paths for openssl-sys build script
              export OPENSSL_DIR="${pkgs.openssl.out}"
              export OPENSSL_LIB_DIR="${pkgs.openssl.out}/lib"
              export OPENSSL_INCLUDE_DIR="${pkgs.openssl.dev}/include"
              # Ensure pkg-config can find the openssl .pc file
              export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.pkg-config}/lib/pkgconfig:$PKG_CONFIG_PATH"
              # Debug: List contents of OpenSSL lib directory to verify
              echo "OpenSSL lib directory contents:"
              ls -la ${pkgs.openssl.out}/lib || echo "Failed to list OpenSSL lib directory"
              echo ">>> OpenSSL environment variables set by shellHook <<<"
            '';
          };

          # Add app for `nix run`
          apps = {
            default = {
              type = "app";
              program = "${rustdocs-mcp-server}/bin/rustdocs_mcp_server";
            };
          };
        };
    };
}
