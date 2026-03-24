root@kc-vse-4-debian:~# lspci | grep Instinct
1e:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Arcturus GL-XL [Instinct MI100] (rev 01)
3f:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Arcturus GL-XL [Instinct MI100] (rev 01)
45:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Arcturus GL-XL [Instinct MI100] (rev 01)
8a:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Arcturus GL-XL [Instinct MI100] (rev 01)
90:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Arcturus GL-XL [Instinct MI100] (rev 01)
b5:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Arcturus GL-XL [Instinct MI100] (rev 01)
Вот, такой вывод даёт команда lspci с адресами GPU (на штатной ЭВМ).
Адреса на PCI-E для сетевых карт Mellanox:
root@kc-vse-4-debian:~# lspci | grep Mellanox
20:00.0 Ethernet controller: Mellanox Technologies MT27800 Family [ConnectX-5]
20:00.1 Ethernet controller: Mellanox Technologies MT27800 Family [ConnectX-5]
40:00.0 Ethernet controller: Mellanox Technologies MT27800 Family [ConnectX-5]
40:00.1 Ethernet controller: Mellanox Technologies MT27800 Family [ConnectX-5]
42:00.0 Ethernet controller: Mellanox Technologies MT27800 Family [ConnectX-5]
42:00.1 Ethernet controller: Mellanox Technologies MT27800 Family [ConnectX-5]
8b:00.0 Ethernet controller: Mellanox Technologies MT27800 Family [ConnectX-5]
8b:00.1 Ethernet controller: Mellanox Technologies MT27800 Family [ConnectX-5]
8d:00.0 Ethernet controller: Mellanox Technologies MT27800 Family [ConnectX-5]
8d:00.1 Ethernet controller: Mellanox Technologies MT27800 Family [ConnectX-5]
b7:00.0 Ethernet controller: Mellanox Technologies MT27800 Family [ConnectX-5]
b7:00.1 Ethernet controller: Mellanox Technologies MT27800 Family [ConnectX-5]
Идея такая, чтобы пользователь как-то мог указать в библиотеке с какой именно видеокартой он хочет работать (предполагаем, что пользователь уже значет, как соотносятся слоты на материнской плате и PCI-E адреса - эту карту соответствий мы сделаем заранее и она не будет меняться в процессе работы).
