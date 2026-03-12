#!/bin/bash
# Deploy da API Brief-Reun na Hostinger VPS com Docker
# Uso: bash deploy/setup.sh
# Pré-requisito: Ubuntu 22.04+, rodar como root ou com sudo

set -e

DOMAIN="api.seudominio.com"   # <-- altere para seu domínio
EMAIL="seu@email.com"         # <-- altere para seu e-mail (Let's Encrypt)

echo "==> Instalando Docker..."
apt-get update
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

echo "==> Verificando .env..."
if [ ! -f ".env" ]; then
    echo "ERRO: arquivo .env não encontrado na raiz do projeto."
    echo "Crie o .env com as variáveis necessárias antes de continuar."
    exit 1
fi

echo "==> Substituindo domínio nos arquivos de config..."
sed -i "s/api.seudominio.com/$DOMAIN/g" deploy/nginx.conf docker-compose.yml
sed -i "s/seu@email.com/$EMAIL/g" docker-compose.yml

echo "==> Subindo Nginx (HTTP) para emissão do certificado SSL..."
docker compose up -d nginx

echo "==> Obtendo certificado SSL (Let's Encrypt)..."
docker compose run --rm certbot

echo "==> Reiniciando Nginx com SSL ativo..."
docker compose restart nginx

echo "==> Subindo toda a aplicação..."
docker compose up -d --build

echo ""
echo "✓ Deploy concluído!"
echo "  API:          https://$DOMAIN"
echo "  Docs Swagger: https://$DOMAIN/docs"
echo "  Health check: https://$DOMAIN/health"
echo ""
echo "Comandos úteis:"
echo "  docker compose logs -f api"
echo "  docker compose ps"
echo "  docker compose restart api"
