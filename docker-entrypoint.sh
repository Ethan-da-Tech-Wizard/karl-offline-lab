#!/bin/sh
# Program: karl-offline-lab - docker-entrypoint.sh
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Container entrypoint that creates a self-signed TLS certificate when the server listens on a non-loopback address, then runs the container command.
# Karl container entrypoint.
#
# KARL_WS_HOST=0.0.0.0 (set by docker-compose.yml / k8s) makes the WSS
# bridge (app/engine/websocket_server.py::_start_server) require a real TLS
# cert at data/ssl/localhost.crt|.key and refuse to fall back to plaintext
# on any non-loopback host. In Kubernetes that cert is provisioned by
# cert-manager (see k8s/certificate.yaml / helm templates/certificate.yaml).
# Plain `docker compose up` has no cert-manager, so generate a self-signed
# cert on first boot if one isn't already present on the mounted data
# volume -- otherwise the bridge silently never binds.
# Stop on the first failing command.
set -e

# Certificate and key locations (the folder can be overridden by an environment variable).
CERT_DIR="${KARL_SSL_DIR:-data/ssl}"
CERT_FILE="$CERT_DIR/localhost.crt"
KEY_FILE="$CERT_DIR/localhost.key"

# Only non-loopback hosts need a certificate; if neither file exists yet, generate one.
if [ "${KARL_WS_HOST:-localhost}" != "localhost" ] && \
   [ "${KARL_WS_HOST:-localhost}" != "127.0.0.1" ] && \
   [ "${KARL_WS_HOST:-localhost}" != "::1" ]; then
    if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
        echo "docker-entrypoint: generating self-signed TLS cert for WSS bridge at $CERT_DIR" >&2
        mkdir -p "$CERT_DIR"
        # Create a 10-year self-signed RSA certificate with no passphrase on the key.
        openssl req -x509 -newkey rsa:2048 -nodes \
            -keyout "$KEY_FILE" -out "$CERT_FILE" \
            -days 3650 -subj "/CN=karl-backend" \
            -addext "subjectAltName=DNS:karl-backend,DNS:localhost" \
            2>/dev/null
        # Keep the private key readable only by its owner.
        chmod 600 "$KEY_FILE"
    fi
fi

# Replace this shell with the real command so it receives signals directly.
exec "$@"
