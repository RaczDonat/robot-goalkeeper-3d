# Szakdolgozat Dokumentáció Vázlat és Irányelvek
## Valós idejű 3D labdadetektálás és robotkapus vezérlés

Ez a dokumentum a debreceni egyetem Mérnökinformatikus BSc szakdolgozatának felépítését és fejezeteinek vázlatát tartalmazza magyar nyelven. Ebből a vázlatból közvetlenül felépíthető és megírható a minimum 40 oldalas Word dokumentum.

---

### Szakdolgozati Formai Követelmények (DE IK)
* **Betűtípus:** Times New Roman 12pt (szövegtörzs)
* **Sorköz:** 1,5 sorköz, sorkizárt igazítás
* **Margók:** Bal: 3,5 cm (kötési margó miatt), Jobb: 2,5 cm, Alsó/Felső: 2,5 cm
* **Fejezetek:** Decimális számozás (pl. 1., 1.1., 1.1.1.)
* **Hivatkozások:** IEEE vagy Harvard stílusban, a szövegben jelölve (pl. `[1]`)
* **Ábrák és táblázatok:** Minden ábrának és táblázatnak legyen egyedi száma és aláírása/címe, és a szövegtörzsben kötelező rájuk hivatkozni.

---

# Szakdolgozat Tervezett Fejezetstruktúrája

## 1. Bevezetés
* **Cél:** A téma bemutatása, aktualitása, a robotkapus projektek jelentősége az iparban és az oktatásban (pl. mechatronika, számítógépes látás, valós idejű rendszerek).
* **Projekt Célkitűzése:** Egy kézzel fogható (fizikai) tesztpad létrehozása, amely képes egy gurított/dobott labda megállítására egy kapuvonal mentén.
* **Munkamegosztás (Morvai Roland & Rácz Donát):**
  * *Morvai Roland:* Képfeldolgozás, kamerakezelés (MindVision SDK), sztereó kalibráció és 3D rekonstrukció, AI alapú detektálás Hailo-8L-en.
  * *Rácz Donát:* Pályagörbe becslés, mikrokontrolleres vezérlés, fizikai mechanika építése, motorvezérlés és soros kommunikáció.

## 2. Irodalmi Áttekintés és Elméleti Alapok
* **Számítógépes Látás:** A 2D képalkotástól a 3D rekonstrukcióig. Sztereó kamerarendszerek működési elve (Epipoláris geometria, trianguláció).
* **Objektumdetektálás:** Hagyományos színküszöbölés (OpenCV HSV alapú detektálás) vs modern mély tanulási módszerek (YOLO architektúrák).
* **Valós Idejű Rendszerek:** Késleltetési források (kamera expozíció, USB átviteli idő, kép kicsomagolás, feldolgozás, kommunikációs jitter).
* **Vezérléselmélet:** Trajektória-becslés (fizikai szűrők, pl. Kálmán-szűrő vagy ballisztikus modell) és a beavatkozó szervek vezérlése (PID vezérlés, léptetőmotorok/szervomotorok).

## 3. Rendszerspecifikáció és Hardver Architektúra
* **Vezérlődoboz és Tápellátás:**
  * 5V-os ipari tápegység méretezése (a Raspberry Pi 5 és a perifériák áramfelvételéhez legalább 5A szükséges).
  * Biztonsági és szakszerű szerelési szempontok (földelés, zavarszűrés, ventilátoros hűtés).
* **Számítási Egység:** Raspberry Pi 5 + AI Hat (Hailo-8L, 13 TOPS).
* **Kamera Rendszer:** 
  * 2 db MindVision MC023CG-SY-UB kamera (2.3 Megapixel, Global Shutter, USB3.0).
  * Global Shutter jelentősége: a gördülő zár (rolling shutter) okozta képtorzulás (jello effect) kiküszöbölése gyorsan mozgó labdák esetén.
  * Objektívek gyújtótávolságának megválasztása a látómező (FOV) optimalizálásához.
  * Adatátvitel: EP-USB3HybridcableU-20 aktív optikai kábelek (zavartalan nagysebességű átvitel 20 méteren).
  * Szinkronizáció: CBL-702-8P-SYNC-5M0 kábelek a két kamera hardveres triggereléséhez (sztereó látásnál kritikus, hogy a két kép pontosan ugyanabban az időszeletben készüljön).

## 4. Szoftveres Architektúra és Optimalizáció
* **A Raspberry Pi 5 Teljesítményproblémájának Elemzése:**
  * Miért lassult be a tesztkód? (OpenCV alapértelmezett V4L2 backend lassúsága, szoftveres demosaicing/Bayer konverzió a Pi CPU-ján, egy szálon futó I/O és feldolgozás).
* **Optimalizációs Megoldások:**
  * *MindVision SDK Integráció:* Direct Memory Access (DMA) használata, hardver-közeli beállítások (expozíció, gain, pixelformátum).
  * *Többszálas Programozás (Multithreading/Multiprocessing):* Külön végrehajtási szálak a kamerák képkockáinak fogadására (Frame Reader Threads) és egy külön szál a feldolgozásra/megjelenítésre. Double buffering technika.
  * *NPU Gyorsítás:* YOLOv8-nano modell exportálása Hailo HEF formátumba. Az AI Hat (Hailo-8L) használata a labdadetektálásra, amivel a CPU terhelése minimálisra csökken.
  * *Felbontás és ROI (Region of Interest) optimalizáció:* Csak a játéktér releváns részének beolvasása és feldolgozása a pixel-adatmennyiség csökkentése érdekében.

## 5. 3D Labdadetektálás és Sztereó Látás
* **Kamera Kalibráció:** Sakktábla mintás kalibráció, belső (intrinsic) és külső (extrinsic) kameraparaméterek meghatározása.
* **Rektifikáció:** Képtorzítások eltávolítása és a sztereó képpárok sorba rendezése.
* **2D Detektálás:** Labda szegmentálása a Hailo NPU-val vagy optimalizált szín alapú szegmentációval.
* **Trianguláció:** 2D pixelkoordinátákból (x1, y1) és (x2, y2) a 3D világkoordináták (X, Y, Z) kiszámítása a kamerák közötti bázistávolság alapján.

## 6. Pályagörbe-becslés és Robot Vezérlés
* **Pályagörbe Modellezése:** Gravitáció, légellenállás figyelembevétele. A labda mozgásegyenletei 3D térben.
* **Kapuvonallal Való Metszéspont Kiszámítása:** A Z-tengely menti elmozdulás alapján predikció arra, hogy a labda melyik (X, Y) koordinátán fogja átlépni a kapuvonalat és mikor.
* **Kommunikáció:** Soros porti kommunikációs protokoll (UART / USB CDC) a Raspberry Pi 5 és a robot vezérlőegysége (pl. STM32, Arduino vagy ESP32) között.
* **Beavatkozó Egység:** Léptetőmotorok vagy szervomotorok meghajtása, gyorsulási és lassulási profilok tervezése (S-curve), pozicionálás minimalizált túllövéssel.

## 7. Mérési Eredmények és Értékelés
* **Képkockasebesség (FPS) és Késleltetés (Latency) vizsgálata:** Különböző felbontások és optimalizációs szintek mellett.
* **Detektálási pontosság:** Hibaarány vizsgálata különböző labdasebességek mellett.
* **Kapus védési hatékonysága:** Hány százalékban sikerült a labdákat sikeresen hárítani.

## 8. Összefoglalás és Jövőbeli Tervek
* Elért eredmények összegzése.
* Továbbfejlesztési lehetőségek (pl. csavart labdák röppályájának becslése, intelligensebb védési stratégiák).

---

# Aktuális Fejlesztési Feladatok (Fókuszban)
Jelenleg az **1. fázis** (Labdadetektálás és sztereó követés) lezárult a PC oldalon:
1. **Bilingvális vázlat és Git alapok** beállítása. [x]
2. **Kamerakezelési absztrakció** szimulátorral és OpenCV/MindVision támogatással. [x]
3. **HSV kalibráció és mentés** trackbar panellel. [x]
4. **YOLOv8 sztereó batch optimalizáció** kifejlesztése. [x]
5. **Kétoldali tesztelés loopback kapcsolaton** sikeresen elvégezve. [x]

---

# Melléklet: 1. Fázis Részletes Technikai Leírása (Szakdolgozathoz)

## Fehér focilabda (5-ös méret) HSV kalibrációja
A fehér focilabda detektálása HSV színtérben különleges megközelítést igényel. Mivel a fehér fénynek nincs markáns színezettsége (Hue) és telítettsége (Saturation), de rendkívül világos (high Value), a színszűrést az alábbi elvek szerint állítottuk be alapértelmezetten:
* **Hue (Színezettség):** `0 - 180` (A teljes tartományt lefedjük, mivel a fehér labdán a környezeti fények és a panelek mintái miatt bármilyen színárnyalat minimálisan megjelenhet).
* **Saturation (Telítettség):** `0 - 60` (Alacsony telítettség, ami kiszűri az intenzív színes objektumokat a háttérből).
* **Value (Fényerő):** `180 - 255` (Magas fényerőküszöb, ami biztosítja, hogy csak a kifejezetten világító/fehér felületeket detektálja a rendszer).

Ez az alapbeállítás a `--calibrate` móddal finomhangolható, és a `s` billentyűvel közvetlenül menthető a [config/system_config.yaml](file:///d:/Szakdolgozat/robot-goalkeeper-projekt/config/system_config.yaml) állományba.

## YOLOv8 sztereó-követés optimalizációja (Parallel Batching)
A hagyományos detektálások során a bal és jobb oldali kamerák képkockáit egymás után küldik el a neurális hálózatnak (Sequential Inference):
$$\text{Time}_{\text{total}} = \text{Inference}(\text{Frame}_{\text{left}}) + \text{Inference}(\text{Frame}_{\text{right}})$$

A projektünkben bevezetett **Parallel Batching** optimalizáció segítségével a bal és a jobb oldali képeket egyetlen tömbbe (batch) fogjuk össze, és egyszerre küldjük át az AI modellnek:
$$\text{Time}_{\text{total}} = \text{Inference}([\text{Frame}_{\text{left}}, \text{Frame}_{\text{right}}])$$

Ez a mélytanulási keretrendszerek (pl. PyTorch/TensorRT) belső párhuzamosítási képességeit és a GPU/NPU hardveres magjait használja ki, így a két kamera képének feldolgozási ideje akár **40-45%-kal is csökkenhet** a szekvenciális végrehajtáshoz képest, ami kritikus a real-time (60+ FPS) követés eléréséhez.

