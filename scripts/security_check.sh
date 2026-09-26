#!/bin/bash
echo "============================================"
echo "  SECURITY CHECK - Projet 7 Demonstrateur"
echo "============================================"
echo ""

echo "--- Fichiers de données (ne doivent PAS être dans le git) ---"
found=0
for ext in csv parquet pkl h5; do
    files=$(find . -name "*.$ext" -not -path "./venv/*" -not -path "./.git/*" 2>/dev/null)
    if [ -n "$files" ]; then
        echo "$files" | while read f; do echo "[ALERTE] $f"; done
        found=1
    fi
done
if [ "$found" -eq 0 ]; then
    echo "OK - Aucun fichier de données trouvé."
fi
echo ""

echo "--- Fichiers .env ---"
if ls .env .env.* 2>/dev/null 1>&2; then
    echo "[ALERTE] Fichier(s) .env trouvé(s) !"
else
    echo "OK - Aucun fichier .env."
fi
echo ""

echo "--- Fichiers volumineux (>10 Mo) ---"
large=$(find . -type f -size +10M -not -path "./venv/*" -not -path "./.git/*" 2>/dev/null)
if [ -n "$large" ]; then
    echo "$large" | while read f; do
        size=$(du -h "$f" | cut -f1)
        echo "[ALERTE] $f ($size)"
    done
else
    echo "OK - Aucun fichier > 10 Mo."
fi
echo ""

echo "--- Dossier venv ---"
if [ -d "venv" ]; then
    echo "[ALERTE] Le dossier venv/ est présent, vérifiez qu'il est dans .gitignore"
else
    echo "OK - Pas de dossier venv."
fi
echo ""

echo "--- Chemins absolus dans le code ---"
abs=$(grep -rn "C:\\\\Users\|/home/\|/Users/" --include="*.py" . 2>/dev/null)
if [ -n "$abs" ]; then
    echo "$abs" | while read line; do echo "[ALERTE] $line"; done
else
    echo "OK - Aucun chemin absolu détecté."
fi
echo ""

echo "--- Clés API / tokens ---"
keys=$(grep -rni "api_key\|secret_key\|password\|token=" --include="*.py" . 2>/dev/null | grep -v "session_state\|st\.text_input")
if [ -n "$keys" ]; then
    echo "$keys" | while read line; do echo "[ALERTE] $line"; done
else
    echo "OK - Aucune clé API détectée."
fi
echo ""

echo "--- Git status ---"
git status
echo ""

echo "============================================"
echo "  Si aucune ALERTE, vous pouvez push."
echo "============================================"
