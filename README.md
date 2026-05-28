# Valós idejű 3D labdadetektálás és robotkapus vezérlés
## Real-Time 3D Ball Detection and Robot Goalkeeper Control

Ez a repozitórium a debreceni egyetem Informatikai Karának Mérnökinformatikus szakán készülő szakdolgozati projekt forráskódját és dokumentációját tartalmazza.

This repository contains the source code and documentation for a thesis project at the University of Debrecen, Faculty of Informatics, for the Computer Engineering BSc program.

---

### Projekt Adatok / Project Details
* **Szakdolgozat Címe / Thesis Title:** Valós idejű 3D labdadetektálás és robotkapus vezérlés / Real-Time 3D Ball Detection and Robot Goalkeeper Control
* **Készítők / Authors:**
  * Morvai Roland (Mérnökinformatikus BSc)
  * Rácz Donát (Mérnökinformatikus BSc)
* **Intézmény / Institution:** Debreceni Egyetem, Informatikai Kar / University of Debrecen, Faculty of Informatics
* **Év / Year:** 2026

---

## Tartalomjegyzék / Table of Contents
1. [Magyar Leírás](#magyar-leírás)
2. [English Description](#english-description)
3. [Hardver Specifikáció / Hardware Specification](#hardver-specifikáció--hardware-specification)
4. [Mappaszerkezet / Directory Structure](#mappaszerkezet--directory-structure)
5. [Telepítés és Futtatás / Installation and Running](#telepítés-és-futtatás--installation-and-running)

---

## Magyar Leírás

A projekt célja egy valós idejű, kézzel fogható robotkapus rendszer kifejlesztése. A rendszer két fő pillére a kamerás megfigyelés és képfeldolgozás (3D labdadetektálás sztereó látás és gépi tanulás segítségével), valamint a robotkapus valós idejű vezérlése (pályagörbe-becslés és gyors reakciójú fizikai beavatkozás).

### Főbb fázisok:
1. **Labdadetektálás (Képfeldolgozás):** MindVision ipari kamerák képeinek feldolgozása, a labda 2D pozíciójának detektálása a képkockákon alacsony késleltetéssel.
2. **Sztereó Látás és 3D Rekonstrukció:** A két kamera képének kalibrációja és sztereó képfeldolgozása a labda 3D koordinátáinak meghatározásához.
3. **Pályagörbe Becslés (Predikció):** A labda 3D-s pályájának kiszámítása fizikai modellek segítségével, a kapuvonalat metsző pont előrejelzése.
4. **Vezérlés és Beavatkozás:** A számított metszéspont alapján a kapu vonalán mozgó robotkapus mechanizmus pozicionálása mikrosekundumos nagyságrendű reakcióidővel.

Detailed chapters for the thesis in Hungarian can be found in the [docs/hu/](file:///d:/Szakdolgozat/robot-goalkeeper-projekt/docs/hu) folder.

---

## English Description

The goal of this project is to develop a real-time, physical robot goalkeeper system. The system relies on two main pillars: camera surveillance and image processing (3D ball detection using stereo vision and machine learning), and real-time control of the robot goalkeeper (trajectory prediction and fast-response physical positioning).

### Key Phases:
1. **Ball Detection (Image Processing):** Processing images from MindVision industrial cameras to detect the 2D position of the ball with minimal latency.
2. **Stereo Vision and 3D Reconstruction:** Calibration and stereo processing of the two camera streams to determine the 3D coordinates of the ball in space.
3. **Trajectory Estimation (Prediction):** Predicting the 3D trajectory of the ball using physical models to forecast its intersection point with the goal line.
4. **Control and Actuation:** Positioning the physical goalkeeper mechanism along the goal line based on the calculated intersection point with microsecond-level response times.

Detailed chapters for the thesis in English can be found in the [docs/en/](file:///d:/Szakdolgozat/robot-goalkeeper-projekt/docs/en) folder.

---

## Hardver Specifikáció / Hardware Specification

| Eszköz / Component | Részletek / Details |
| :--- | :--- |
| **Számítógép / Controller unit** | Raspberry Pi 5 (8GB RAM) |
| **AI Gyorsító / AI Accelerator** | Raspberry Pi AI Hat (Hailo-8L NPU, 13 TOPS) |
| **Kamerák / Cameras** | 2x MC023CG-SY-UB (MindVision 2.3MP USB3.0 Global Shutter) |
| **Adatkábelek / Data Cables** | 2x EP-USB3HybridcableU-20 |
| **Szinkronkábelek / Sync Cables** | 2x CBL-702-8P-SYNC-5M0 |
| **Tápegység / Power Supply** | 5V DC Industrial PSU (szerelődobozba építve / integrated in control box) |
| **Objektívek / Lenses** | 2x Industrial lenses |

---

## Mappaszerkezet / Directory Structure

```
d:\Szakdolgozat\robot-goalkeeper-projekt/
├── .gitignore          # Git mellőzési szabályok
├── README.md           # Fő projektismertető (ez a fájl)
├── LICENSE            # MIT Licenc
├── docs/               # Szakdolgozat fejezetek és dokumentáció
│   ├── hu/             # Magyar nyelvű dokumentáció
│   │   └── README.md   # Magyar fejezetek tartalomjegyzéke és vázlata
│   └── en/             # Angol nyelvű dokumentáció
│       └── README.md   # Angol fejezetek tartalomjegyzéke és vázlata
├── src/                # Forráskód (Source code)
│   ├── common/         # Közös modulok (naplózás, időzítés mérők)
│   ├── detection/      # Labda detektálás (NPU / OpenCV alapú)
│   ├── stereo/         # Kamerák kalibrációja és 3D rekonstrukció
│   └── control/        # Robotkapus fizikai vezérlése (soros port)
├── config/             # Kamera kalibrációs és rendszer-beállítások
└── scripts/            # Telepítő és konfigurációs scriptek RPi5-höz
```

---

## Telepítés és Futtatás / Installation and Running

*(Kidolgozás alatt a fejlesztési fázisok előrehaladtával / Under construction as development progresses)*
- Lásd: [docs/hu/README.md](file:///d:/Szakdolgozat/robot-goalkeeper-projekt/docs/hu/README.md) a részletes beállításokért.
- See: [docs/en/README.md](file:///d:/Szakdolgozat/robot-goalkeeper-projekt/docs/en/README.md) for detailed English instructions.
