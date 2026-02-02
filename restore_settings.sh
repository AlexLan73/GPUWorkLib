#!/bin/bash
# Восстановление настроек на Ubuntu машине
# Запуск: bash restore_settings.sh

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}🔄 Восстановление DrvGPU настроек на Ubuntu...${NC}\n"

# Найти архив
BACKUP_FILE=$(ls DrvGPU-Settings-*.zip 2>/dev/null | sort -r | head -1)

if [ -z "$BACKUP_FILE" ]; then
    echo -e "${RED}❌ Backup файл DrvGPU-Settings-*.zip не найден!${NC}"
    echo -e "${YELLOW}Поместите архив в текущую папку проекта и повторите.${NC}"
    exit 1
fi

echo -e "${GREEN}📦 Найден архив: $BACKUP_FILE${NC}"
echo -e "${YELLOW}⏳ Распаковываю...${NC}\n"

# Проверить наличие unzip
if ! command -v unzip &> /dev/null; then
    echo -e "${RED}❌ unzip не установлен!${NC}"
    echo -e "${YELLOW}Установи: sudo apt install unzip${NC}"
    exit 1
fi

# Распаковать архив
if unzip -o "$BACKUP_FILE" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Успешно восстановлено!${NC}"
    echo -e "${GREEN}   Файлы:${NC}"
    echo -e "${GREEN}   ✓ .vscode/ (настройки VSCode)${NC}"
    echo -e "${GREEN}   ✓ cmake/ (модули CMake)${NC}"
    echo -e "${GREEN}   ✓ CMakeLists.txt (все уровни)${NC}"
    echo -e "\n${GREEN}🚀 Готово! Открой проект в VSCode:${NC}"
    echo -e "${YELLOW}   code .${NC}\n"
else
    echo -e "${RED}❌ Ошибка при распаковке архива${NC}"
    exit 1
fi
