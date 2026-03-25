Отличный вопрос — это стандартная задача с multi-remote Git. Разберём все варианты.Разберём каждую стратегию пошагово.

---

## Стратегия 1: Два отдельных remote (базовая)

Самый простой и понятный вариант — каждый сервер получает свой именованный remote:

```bash
# Начальная настройка
git remote add internal git@your-local-server.lan:team/my-project.git
git remote add github   git@github.com:yourorg/my-project.git

# Проверяем
git remote -v
# internal  git@your-local-server.lan:team/my-project.git (fetch)
# internal  git@your-local-server.lan:team/my-project.git (push)
# github    git@github.com:yourorg/my-project.git (fetch)
# github    git@github.com:yourorg/my-project.git (push)

# Push — две отдельные команды
git push internal main
git push github main
```

Это работает, но нужно каждый раз набирать push дважды, что неудобно и легко забыть один из них.

---

## Стратегия 2: Multi-push URL на одном remote (рекомендую)

Git поддерживает несколько push URL на одном remote, и при обычном `git push` он отправляет во все зарегистрированные URL одновременно. Это идеальный вариант для вас:

```bash
# Допустим, у вас уже есть origin на локальный сервер
git remote add origin git@your-local-server.lan:team/my-project.git

# Добавляем GitHub как ВТОРОЙ push URL к тому же origin
# ВАЖНО: первый --add перезаписывает дефолтный push URL,
# поэтому нужно добавить ОБА URL
git remote set-url --add --push origin git@your-local-server.lan:team/my-project.git
git remote set-url --add --push origin git@github.com:yourorg/my-project.git

# Проверяем — fetch один, push два:
git remote -v
# origin  git@your-local-server.lan:team/my-project.git (fetch)
# origin  git@your-local-server.lan:team/my-project.git (push)
# origin  git@github.com:yourorg/my-project.git (push)
```

Теперь одна команда отправляет в оба места:

```bash
git push origin main          # → локальный + GitHub
git push origin feature/auth  # → оба
git push origin --tags        # → теги туда и туда
git push                      # если tracking настроен — тоже оба
```

Fetch при этом идёт только с одного сервера (первый URL), что логично — вы работаете с одним "источником правды".

Вот как это выглядит в `.git/config`:

```ini
[remote "origin"]
    url = git@your-local-server.lan:team/my-project.git
    fetch = +refs/heads/*:refs/remotes/origin/*
    pushurl = git@your-local-server.lan:team/my-project.git
    pushurl = git@github.com:yourorg/my-project.git
```

---

## Стратегия 3: Отдельные remote + виртуальный "all" (гибкая)

Можно создать отдельный remote `all`, который используется только для push в оба места одновременно, сохранив при этом возможность пушить и пуллить с каждого сервера по отдельности:

```bash
# Именованные remote для индивидуального доступа
git remote add internal git@your-local-server.lan:team/my-project.git
git remote add github   git@github.com:yourorg/my-project.git

# Виртуальный "all" для одновременного push
git remote add all git@your-local-server.lan:team/my-project.git
git remote set-url --add --push all git@your-local-server.lan:team/my-project.git
git remote set-url --add --push all git@github.com:yourorg/my-project.git
```

Теперь у вас три опции:

```bash
git push all main       # → оба сервера
git push internal main  # → только локальный
git push github main    # → только GitHub

git pull internal main  # fetch + merge с локального
git pull github main    # fetch + merge с GitHub
```

Это самый гибкий вариант — пригождается когда нужно точечно тянуть изменения с разных серверов или пушить выборочно.

---

## Продвинутые сценарии

**Разные ветки в разные remote** (например, `dev` → локальный, `release/*` → GitHub):

```bash
# Настраиваем tracking по умолчанию
git branch --set-upstream-to=internal/main main
git branch --set-upstream-to=github/main release/v2

# Или через git config напрямую
git config branch.main.remote internal
git config branch.main.merge refs/heads/main
```

**Git hook — автоматический push во второй remote** (если не хотите менять push URL):

```bash
# .git/hooks/post-push не существует, но можно использовать post-commit
# или лучше — git alias:

git config alias.pushall '!git push internal && git push github'

# Теперь:
git pushall main
```

**Скрипт для всех remote разом** (если их больше двух):

```bash
# ~/.bashrc или ~/.zshrc
git-pushall() {
  local branch=${1:-$(git rev-parse --abbrev-ref HEAD)}
  for remote in $(git remote); do
    echo "→ Pushing to $remote..."
    git push "$remote" "$branch"
  done
}
```

**Защита от случайного push на GitHub (фильтр по веткам)**:

```ini
# .git/config — push только определённые ветки на GitHub
[remote "github"]
    url = git@github.com:yourorg/my-project.git
    fetch = +refs/heads/*:refs/remotes/github/*
    push = refs/heads/main:refs/heads/main
    push = refs/heads/release/*:refs/heads/release/*
    # dev, feature/* и т.д. НЕ попадут на GitHub
```

---

## Моя рекомендация для вашего случая

Учитывая, что у вас большой проект с командой:

**Стратегия 3 (отдельные remote + "all")** — потому что:

1. Локальный сервер — основной "источник правды" для отдела, с него команда делает `pull`
2. GitHub — зеркало или внешний доступ, туда push идёт параллельно
3. Нужна возможность точечно тянуть что-то только с одного из серверов

```bash
# Разовая настройка для каждого разработчика:
git remote add internal git@local-git.company.lan:team/my-project.git
git remote add github   git@github.com:yourorg/my-project.git
git remote add all      git@local-git.company.lan:team/my-project.git
git remote set-url --add --push all git@local-git.company.lan:team/my-project.git
git remote set-url --add --push all git@github.com:yourorg/my-project.git

# Ежедневная работа:
git pull internal main    # тянем с основного
git push all main         # пушим в оба
```

А если хотите, чтобы эта конфигурация была частью проекта (чтобы каждому новому разработчику не настраивать руками), можно добавить setup-скрипт в репозиторий — хотите, сгенерю?