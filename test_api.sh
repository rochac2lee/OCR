#!/bin/bash
# Script de teste da API de detecção de números de camisas

API_URL="http://localhost:8000"

echo "=========================================="
echo "Testando API de Detecção de Números"
echo "=========================================="
echo ""

# Teste 1: Health Check
echo "1️⃣  Testando Health Check..."
response=$(curl -s -w "\n%{http_code}" "${API_URL}/")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" -eq 200 ]; then
    echo "✅ Health Check OK"
    echo "   Resposta: $body"
else
    echo "❌ Health Check FALHOU (HTTP $http_code)"
    echo "   Resposta: $body"
    exit 1
fi
echo ""

# Teste 2: Enviar imagem (precisa de uma imagem de teste)
echo "2️⃣  Testando detecção de números..."
if [ ! -f "test_image.jpg" ]; then
    echo "⚠️  Arquivo test_image.jpg não encontrado"
    echo "   Para testar a detecção, crie um arquivo test_image.jpg"
    echo "   Exemplo: curl -X POST -F 'image=@sua_imagem.jpg' ${API_URL}/predict"
else
    response=$(curl -s -w "\n%{http_code}" -X POST -F "image=@test_image.jpg" "${API_URL}/predict")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" -eq 200 ]; then
        echo "✅ Detecção executada com sucesso"
        echo "   Resposta:"
        echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
    else
        echo "❌ Detecção FALHOU (HTTP $http_code)"
        echo "   Resposta: $body"
    fi
fi
echo ""

# Teste 3: Erro - sem imagem
echo "3️⃣  Testando validação (sem imagem)..."
response=$(curl -s -w "\n%{http_code}" -X POST "${API_URL}/predict")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" -eq 400 ]; then
    echo "✅ Validação funcionando corretamente"
    echo "   Resposta: $body"
else
    echo "⚠️  Resposta inesperada (HTTP $http_code)"
    echo "   Resposta: $body"
fi
echo ""

echo "=========================================="
echo "Testes concluídos!"
echo "=========================================="

