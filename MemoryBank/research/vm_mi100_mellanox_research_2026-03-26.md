# Исследование: Разработка под MI100 + Mellanox в виртуальной машине

**Дата**: 2026-03-26
**Автор**: Кодо (AI-ассистент GPUWorkLib)
**Заказчик**: Alex
**Задача**: Оценить возможность разработки под AMD Instinct MI100 и Mellanox ConnectX в VM (Astra Linux 1.8 / Debian 13)

---

## 1. Постановка задачи

Необходимо определить, можно ли вести разработку проектов, использующих:
- **AMD Instinct MI100** (gfx908, CDNA1) — GPU-вычисления через ROCm/HIP
- **Mellanox ConnectX** — InfiniBand/RDMA сетевые операции

при установке **Astra Linux 1.8** или **Debian 13** в качестве гостевой ОС на виртуальную машину.

Рассматриваемые гипервизоры: **VirtualBox**, **VMware Workstation**, а также альтернативы.

---

## 2. Сводная таблица результатов

| Гипервизор / Подход | GPU Passthrough (MI100) | Mellanox (SR-IOV / RDMA) | Пригодность |
|---|---|---|---|
| **VirtualBox** | ❌ Невозможен | ❌ Нет SR-IOV | Только компиляция |
| **VMware Workstation** | ❌ Невозможен | ❌ Нет SR-IOV | Только компиляция |
| **VMware ESXi** | ⚠️ DirectPath I/O (MI100 не протестирован) | ✅ SR-IOV | Bare-metal гипервизор |
| **KVM/QEMU + VFIO** | ✅ Работает (подтверждено) | ✅ SR-IOV | **Рекомендуется** |
| **Docker / LXC контейнер** | ✅ Напрямую (`--device`) | ✅ Host network | **Проще всего** |
| **chroot / systemd-nspawn** | ✅ Напрямую | ✅ Напрямую | Минимальная изоляция |
| **Dual Boot** | ✅ Нативно | ✅ Нативно | Максимальная производительность |

---

## 3. Детальный анализ гипервизоров

### 3.1. VirtualBox — ❌ Не подходит

- **PCIe passthrough удалён начиная с версии 6.1** (ранее был экспериментальным, только PCI, только Linux-хост)
- SR-IOV не поддерживается
- GPU-вычисления из гостевой ОС невозможны
- **Максимум**: установить ROCm SDK (hip-dev, hipcc) + Mellanox OFED headers → компилировать код, но без запуска на реальном оборудовании

**Источник**: [Oracle VirtualBox PCI Passthrough Documentation](https://docs.oracle.com/en/virtualization/virtualbox/6.0/admin/pcipassthrough.html), [VirtualBox Forum — State of PCIe Passthrough](https://forums.virtualbox.org/viewtopic.php?t=83661)

### 3.2. VMware Workstation — ❌ Не подходит

- Desktop-версия VMware (Workstation / Player) **не поддерживает GPU passthrough**
- DirectPath I/O — функция исключительно **VMware ESXi** (bare-metal гипервизор)
- SR-IOV для Mellanox также не поддерживается в desktop-версии
- **Максимум**: аналогично VirtualBox — только компиляция без запуска

**Источник**: [VMware DirectPath I/O Blog](https://blogs.vmware.com/cloud-foundation/2018/09/11/using-gpus-with-virtual-machines-on-vsphere-part-2-vmdirectpath-i-o/)

### 3.3. VMware ESXi — ⚠️ Теоретически возможно

- Поддерживает DirectPath I/O (PCIe passthrough) для GPU
- Mellanox SR-IOV поддерживается (RoCE, InfiniBand)
- **Однако**: MI100 официально не протестирован. Есть сообщения о проблемах с AMD Instinct MI60 на ESXi 8 (issue ROCm/ROCm#4017)
- Требуется bare-metal установка ESXi (это не desktop-гипервизор)
- Производительность: ~95-96% от нативной

**Источник**: [ROCm Issue #4017 — MI60 ESXi](https://github.com/ROCm/ROCm/issues/4017), [VMware RoCE SR-IOV Setup](https://www.vmware.com/docs/vsphere7x-roce-sriov-setup-perf)

### 3.4. KVM/QEMU + VFIO — ✅ Рекомендуемый вариант для VM

- **GPU passthrough MI100 подтверждён** пользователями (Proxmox/KVM + Ubuntu 22.04 гость, ROCm 6.0+)
- Mellanox ConnectX: SR-IOV работает (VF проброс в гостевую ОС)
- Инструменты: `virt-manager` (GUI), `virsh`, QEMU CLI

**Требования**:
- Linux-хост с поддержкой IOMMU (AMD-Vi в BIOS)
- Модули ядра: `vfio`, `vfio-pci`, `vfio-iommu-type1`
- MI100 и контроллер должны быть в отдельной IOMMU-группе (или использовать ACS override patch)
- Для Mellanox: прошить firmware с VF > 0 (`mlxconfig`)

**Ограничения**:
- Infinity Fabric Bridge между MI100 не работает в VM (GPU взаимодействуют только через PCIe)
- SR-IOV через AMD GIM driver для MI100 **не поддерживается** (GIM поддерживает MI300X+, MI350/355). Только полный passthrough.
- Один MI100 = одна VM (нельзя шарить между несколькими VM)

**Источники**:
- [PyTorch Issue #115734 — MI100 in VM](https://github.com/pytorch/pytorch/issues/115734)
- [ROCm Issue #2722 — MI100 Bridge in VM](https://github.com/ROCm/ROCm/issues/2722)
- [Mellanox SR-IOV Configuration Guide](https://enterprise-support.nvidia.com/s/article/HowTo-Configure-SR-IOV-for-ConnectX-4-ConnectX-5-ConnectX-6-with-KVM-Ethernet)
- [Gentoo Wiki — GPU Passthrough KVM](https://wiki.gentoo.org/wiki/GPU_passthrough_with_virt-manager,_QEMU,_and_KVM)

---

## 4. Совместимость ОС с ROCm (для MI100)

По официальной документации ROCm 7.2:

| Гостевая ОС | ROCm + MI100 | Примечание |
|---|---|---|
| **Ubuntu 22.04.5** | ✅ Поддерживается | Основной рекомендуемый вариант |
| **Ubuntu 24.04.4** | ✅ Поддерживается | |
| **RHEL 8.10** | ✅ Поддерживается | |
| **RHEL 9.4 / 9.6 / 9.7** | ✅ Поддерживается | |
| **RHEL 10.0 / 10.1** | ✅ Поддерживается | |
| **SLES 15 SP7** | ✅ Поддерживается | |
| **Debian 13** | ❌ Официально нет (только MI300X+) | Практически работает (проверено в GPUWorkLib) |
| **Debian 12** | ❌ Официально нет | |
| **Astra Linux 1.8** | ❌ Не в матрице ROCm | Основана на Debian; ядро может быть старым для amdgpu |

> **Примечание**: Несмотря на отсутствие MI100 в официальной матрице для Debian 13, проект GPUWorkLib успешно работает на этой связке. Официальная матрица — рекомендация AMD, а не жёсткое техническое ограничение.

> **Astra Linux 1.8**: Основана на Debian. ROCm может заработать при совпадении версий ядра и glibc, но это неподдерживаемая конфигурация. Ключевой риск — версия ядра (нужен 5.15+ для amdgpu с MI100).

**Источник**: [ROCm System Requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

---

## 5. Альтернативные подходы (без полноценной VM)

### 5.1. Docker / LXC контейнер — рекомендуется при отсутствии потребности в полной ОС

```bash
# Пример запуска контейнера с GPU-доступом
docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -v /opt/rocm:/opt/rocm:ro \
  debian:13 bash
```

- GPU доступен напрямую через устройства хоста
- Mellanox доступен через `--network=host` или проброс устройства
- Минимальные накладные расходы, нет потерь производительности
- Можно создать образ на базе Astra Linux (если есть доступ к репозиторию)

### 5.2. systemd-nspawn / chroot

- Полноценная файловая система Astra Linux / Debian 13 без виртуализации
- Прямой доступ ко всему оборудованию хоста
- Нулевые потери производительности
- Подходит для сборки и тестирования

### 5.3. Dual Boot

- Установить Astra Linux 1.8 / Debian 13 второй системой
- Нативный доступ к MI100 + Mellanox
- Максимальная производительность
- Переключение через перезагрузку

---

## 6. Рекомендации

### Для полноценной разработки с GPU + RDMA в VM:

| Приоритет | Подход | Комментарий |
|---|---|---|
| 🥇 | **KVM/QEMU + VFIO passthrough** | Единственный гипервизор с работающим GPU passthrough для MI100 |
| 🥈 | **Docker контейнер** | Проще в настройке, GPU доступен напрямую, но не полная ОС |
| 🥉 | **Dual Boot** | Максимальная производительность, но неудобство переключения |

### Для компиляции кода без запуска на GPU:

Любая VM (VirtualBox, VMware) подойдёт — установить ROCm SDK + OFED headers.

### НЕ рекомендуется:

- **VirtualBox** для работы с GPU/RDMA — технически невозможно
- **VMware Workstation** для работы с GPU/RDMA — технически невозможно
- **Astra Linux 1.8 как гостевая ОС** для ROCm — высокий риск несовместимости ядра

---

## 7. Схема принятия решения

```
Нужен ли GPU (MI100) из гостевой ОС?
├── НЕТ (только компиляция) → Любая VM (VirtualBox / VMware) ✅
└── ДА (запуск на GPU)
    ├── Нужна полная изоляция ОС?
    │   ├── ДА → KVM/QEMU + VFIO passthrough ✅
    │   └── НЕТ
    │       ├── Нужна файловая система целевой ОС? → Docker / systemd-nspawn ✅
    │       └── Нужна нативная производительность? → Dual Boot ✅
    └── VirtualBox / VMware Workstation → ❌ Невозможно
```

---

## 8. Источники

1. [ROCm System Requirements — Official](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
2. [ROCm Issue #2722 — MI100 Bridge in VM](https://github.com/ROCm/ROCm/issues/2722)
3. [PyTorch Issue #115734 — MI100 GPU in VM](https://github.com/pytorch/pytorch/issues/115734)
4. [Oracle VirtualBox PCI Passthrough](https://docs.oracle.com/en/virtualization/virtualbox/6.0/admin/pcipassthrough.html)
5. [VirtualBox Forum — Passthrough Removed](https://forums.virtualbox.org/viewtopic.php?t=107158)
6. [VMware DirectPath I/O](https://blogs.vmware.com/cloud-foundation/2018/09/11/using-gpus-with-virtual-machines-on-vsphere-part-2-vmdirectpath-i-o/)
7. [ROCm Issue #4017 — MI60 ESXi Failure](https://github.com/ROCm/ROCm/issues/4017)
8. [Mellanox SR-IOV for ConnectX with KVM](https://enterprise-support.nvidia.com/s/article/HowTo-Configure-SR-IOV-for-ConnectX-4-ConnectX-5-ConnectX-6-with-KVM-Ethernet)
9. [VMware RoCE SR-IOV Setup](https://www.vmware.com/docs/vsphere7x-roce-sriov-setup-perf)
10. [AMD GIM Virtualization Driver (GitHub)](https://github.com/amd/MxGPU-Virtualization)
11. [AMD Instinct Virtualization Driver Docs](https://instinct.docs.amd.com/projects/virt-drv/en/latest/index.html)
12. [Gentoo Wiki — GPU Passthrough KVM](https://wiki.gentoo.org/wiki/GPU_passthrough_with_virt-manager,_QEMU,_and_KVM)
13. [Configuring SR-IOV for Mellanox Adapters](https://shawnliu.me/post/configuring-sr-iov-for-mellanox-adapters/)
14. [AMD ROCm Virtualization & Containers](https://rocmdoc.readthedocs.io/en/latest/ROCm_Virtualization_Containers/ROCm-Virtualization-&-Containers.html)
