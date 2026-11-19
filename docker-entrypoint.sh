#!/bin/sh
# ==============================================================================
# Program: docker-entrypoint.sh
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Container initialization and bootstrap script.
#          Configures runtime environment variables, provisions self-signed TLS
#          certificates for secure WebSocket (WSS) communication, and delegates
#          execution to the container command.
# ==============================================================================

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
set -e

# Directory (overridable via KARL_SSL_DIR) that holds the TLS cert the WSS bridge looks for.
CERT_DIR="${KARL_SSL_DIR:-data/ssl}"
CERT_FILE="$CERT_DIR/localhost.crt"
KEY_FILE="$CERT_DIR/localhost.key"

# Only needed when the bridge listens beyond loopback; local-only runs do not require TLS.
if [ "${KARL_WS_HOST:-localhost}" != "localhost" ] && \
   [ "${KARL_WS_HOST:-localhost}" != "127.0.0.1" ] && \
   [ "${KARL_WS_HOST:-localhost}" != "::1" ]; then
    # Never overwrite an existing cert (for example one provisioned by cert-manager or kept on the
    # data volume).
    if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
        echo "docker-entrypoint: generating self-signed TLS cert for WSS bridge at $CERT_DIR" >&2
        mkdir -p "$CERT_DIR"
        # Self-signed 10-year cert with SANs for the compose service name and localhost; adequate for
        # compose, not a public CA cert.
        openssl req -x509 -newkey rsa:2048 -nodes \
            -keyout "$KEY_FILE" -out "$CERT_FILE" \
            -days 3650 -subj "/CN=karl-backend" \
            -addext "subjectAltName=DNS:karl-backend,DNS:localhost" \
            2>/dev/null
        # Private key readable by the owner only.
        chmod 600 "$KEY_FILE"
    fi
fi

# Hand over PID 1 to the container command so it receives signals (docker stop) directly.
exec "$@"
