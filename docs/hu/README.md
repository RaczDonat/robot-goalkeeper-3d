# Szakdolgozat Elméleti és Hardver Fejezetek Vázlata
## Valós idejű 3D labdadetektálás és robotkapus vezérlés

Ez a dokumentum a debreceni egyetem Mérnökinformatikus BSc szakdolgozatának azon fejezeteit és részletes vázlatát tartalmazza, amelyeket **már most, a programozás és tesztelés részletezése előtt** meg tudsz írni. Ez a szakdolgozat első 15-20 oldalának a gerince.

---

# Tervezett Fejezetstruktúra (Elmélet és Hardver Fázis)

## 1. Bevezetés és Rendszerkoncepció
*Ez a fejezet bemutatja a projekt hátterét, motivációját és a célkitűzéseket.*
* **1.1 Előszó és Motiváció:** 
  - A valós idejű rendszerek és a mechatronika jelentősége a mai iparban.
  - Miért izgalmas a robotkapus téma? (Gyors reakcióidő, szenzorfúzió, precíz aktuátor-mozgatás kombinációja).
* **1.2 Témaválasztás és Aktualitás:**
  - Miért fontos a nagysebességű képfeldolgozás? (Autonóm járművek, ipari minőségellenőrzés kapcsolata).
* **1.3 Célkitűzések:**
  - Egy fizikai (kézzel fogható) tesztpad létrehozása, amely képes egy gurított/dobott labda megállítására egy kapuvonal mentén.
  - A rendszer legyen moduláris és alacsony késleltetésű.
* **1.4 Robotok kialakulásának története:**
  - Rövid történelmi áttekintés az ipari robotoktól (pl. Unimate) a modern kollaboratív robotokig (cobotok) és autonóm rendszerekig.
* **1.5 Robotkapus fogalmának tisztázása és szakirodalma:**
  - Milyen létező megoldások vannak? (Pl. a Robokeeper koncepciója, egyetemi kutatóprojektek).
  - Mi a különbség a passzív és az aktív védési mechanizmusok között?
* **1.6 Munkamegosztás:**
  - *Morvai Roland:* Képfeldolgozás, sztereó kalibráció és 3D rekonstrukció, AI alapú detektálás.
  - *Rácz Donát:* Pályagörbe becslés, mikrokontrolleres vezérlés, fizikai mechanika építése, motorvezérlés.

## 2. Elméleti Alapok (Gépi látás és Matematika)
*Ez a fejezet tisztázza a 3D rekonstrukció és a képfeldolgozás matematikai és elméleti hátterét.*
* **2.1 Számítógépes látás (Computer Vision) alapelvei:**
  - A digitális képalkotás folyamata (pixelrácsok, színterek: RGB vs. HSV színtér elmélete, miért előnyösebb a HSV a színszűrésre).
* **2.2 Sztereó látáselmélet:**
  - Hogyan lát az ember és hogyan másolható ez le két kamerával?
  - Epipoláris geometria elmélete (alapvető és lényeges mátrixok fogalma).
* **2.3 Trianguláció és mélységbecslés:**
  - A sztereó képeltérés (Disparity) matematikai definíciója.
  - Hogyan számolunk 3D koordinátát a két kamera pixelkoordinátáiból (hasonló háromszögek elve, bázistávolság és gyújtótávolság összefüggése).
* **2.4 Objektumdetektálási módszerek elmélete:**
  - *Hagyományos módszerek:* Küszöbölés, kontúrkeresés, geometriai kör-illesztés (Hough-transzformáció).
  - *Modern módszerek:* Konvolúciós neurális hálózatok (CNN) működési elve, a YOLO (You Only Look Once) architektúra fejlődése és működése (Bounding box, Confidence score, Class prediction).

## 3. A Rendszer Hardver Architektúrája és Eszközbemutatása
*Ez a fejezet bemutatja a fizikai eszközöket, részletezve a technikai specifikációkat és a választás okait.*
* **3.1 A számítási egység és AI Hat:**
  - **Raspberry Pi 5 (8GB RAM):** Processzor architektúra (BCM2712 Broadcom), teljesítménybeli fejlődés a Pi 4-hez képest, GPIO interfészek.
  - **Raspberry Pi AI Hat (Hailo-8L NPU):** Mi az az NPU (Neural Processing Unit)? 13 TOPS számítási kapacitás jelentősége a mélytanulási modellek futtatásánál.
* **3.2 Ipari kamerák és optikai kiegészítők:**
  - **MindVision MC023CG-SY-UB:** A szenzor tulajdonságai (Sony IMX392), Global Shutter technológia elmélete (miért elengedhetetlen a gyors mozgásoknál a gördülőzár / rolling shutter kiküszöbölésére).
  - **Objektívek:** Gyújtótávolság, blende, látómező (FOV) tervezése.
  - **EP-USB3HybridcableU-20 aktív optikai USB 3.0 kábel:** Miért van szükség aktív kábelre 20 méteren? (Jelcsillapítás és elektromágneses zajvédelem).
  - **CBL-702-8P-SYNC-5M0 szinkronkábel:** Hardveres triggerelés elmélete sztereó látásnál.
* **3.3 Vezérlődoboz és Tápellátás kialakítása:**
  - Az 5V-os ipari tápegység kiválasztásának mérnöki számítása (RPi 5 áramfelvétele, AI Hat terhelése, kamerák áramigénye USB-n keresztül - legalább 5A / 25W igény).
  - Szerelődoboz elrendezése, hűtési és biztonsági (földelési) szempontok.
* **3.4 A robotkapus fizikai és mechanikai felépítése:**
  - Donát által épített mechanika leírása (sínek, fogasszíjak, lineáris vezetők).
  - Léptetőmotorok/szervomotorok és azok vezérlőkártyái (pl. TB6600 vagy TMC drivers).
