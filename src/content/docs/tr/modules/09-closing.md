---
title: "9. Kapanış: Zinciri Geri Sar"
description: "Sekiz gate tersten okunur, eve götürülecek üç karar, ve Pazartesi laptop'ında hâlâ çalışan şey."
---

## Gate değil — geri sarma

> **Zinciri tersten, yüksek sesle, tek nefeste oku.**

Bu blokta hiçbir şey çalışmıyor. Laptop'lar kapanabilir. Bugün her modül bir hatayla açıldı ve o
hatanın kazandırdığı tek şeyle — bir sonraki modülün cevaplamak zorunda kaldığı soruyla — kapandı.
Sondan okuduğunda her adım, bir öncekinin neden yetmediğinin sebebi.

## Zincir, tersten

**8. Güncellik** — Q2 edisyonu **EUR 120** dedi; yanlış değildi, bayattı. Q3'ü yeniden ingest et,
aynı soru **EUR 90** okunuyor: training yok, yeni weight yok, tek bir `POST /ingest`. RAG'in
fine-tuning'e karşı bütün davası bu — ve onu adil yapan bir önceki modül.

**7. Chunking ve gürültü** — fixed-280 doğru dokümanı buldu, yanlış sütunu okudu — 12 Eylül
koşusunda "EUR 13" dedi; sendeki sayı farklı olacak. Dokümanın yapısına göre kesmek K satırını
header'ının altına geri koydu, `[fare classic shorthaul > RULE 2A …]`, ve cevap EUR 90 oldu.
**Doğru chunk, doğru cevap** — bu da RAG'in fine-tuning'i asıl önemli yerde yendiğini kanıtlama
hakkını kazandırdı: veri değiştiğinde.

**6. Container'da ChromaDB** — chunk'ları nihayet *görebildik* ve K satırının chunk'ında header
yoktu. Store, içine bakılabilen bir endpoint'i olan bir container; "retrieval çöp getirdi"yi adresi
olan bir hataya çeviren şey bu.

**5. Basit RAG** — ingest, retrieve, query: cevap kaynağa dayandı, kaynağın adı verildi. Ama ne
getirildiğine bak. Yanlış chunk'tan gelen kaynaklı cevap hâlâ yanlış cevaptır — hem de modelin
haklı olduğundaki ses tonuyla.

**4. Veri değişti** — bütün corpus prompt'a sığdı: 12 Eylül koşusunda soru başına ~21 000 token,
soğukta 60–120 s. Her çeyrek yeniden eğitemezsin, her soruda bütün kitabın parasını ödeyemezsin —
o zaman modele doğru parçayı ver. Retrieval burada, bir maliyet tasarrufu olarak doğdu.

**3. Fine-tuning** — Q2'yi kendi weight'lerinden cevaplayan bir model çalışır — veri değişene
kadar. `UNVERIFIED: kraken-q2 kurulmadı; bayat cevap bir tahmin, bu salonun izlediği bir şey
değil.` Hatanın biçimi için modele gerek yok: weight'ler bir satırın değiştiğini bilemez ve
öğrendikleri satırı kaynak gösteremez.

**2. Sinir ağı nasıl öğrenir** — weight'ler, training verisinin donmuş bir fotoğrafı. Bu da tek
makul soruyu doğurdu: fotoğrafı kendi verimizle yeniden çekebilir miyiz?

**1. Çıplak LLM** — verimizi bilmiyordu ve bilmediğini de bilmiyordu. CLASSIC K iptal cezası
sorulduğunda kendinden emin bir tutar uydurdu — 12 Eylül koşusunda "€50" sınıfından bir rakam.
Öyleyse: bir modelde bilgi *nerede*?

Sekiz gate ve bir kurulum, hiçbiri tanımla açılmadı. **Bütün gün tek bir sayı — EUR 90 — bir
dokümandan alıntılanmış, dokümanın adı verilmiş hâliyle.**

<div class="presenter-note">
Geri sarmayı ayakta yap; slayt yok, projektörde laptop yok. Sekiz satır, her biri bir nefes; geçerken
tahtadaki gate'i göster. Yuvarlamadan, dürüst söylemen gereken iki şey var: fine-tune cümlesi bir
tahmin — <code>kraken-q2</code> hiç kurulmadı — ve bugün gördükleri sayılar tek makinede tek koşu,
yani rakamı değil yönü aktarsınlar. Yaklaşık dört dakika. Gün geç kalmışsa tutacağın kısım burası;
onun yerine "laptop'ında ne kalıyor"u tek cümleye indir.
</div>

## Eve götüreceğin üç karar

**Karakter sayısına değil, dokümanın yapısına göre chunk'la.** Günün hatası hiçbir zaman doküman
değildi — header'ını kaybeden satırdı. Markdown başlıkları, tablo başlıkları, madde numaraları
bedava context; sabit boyutlu bir kesici bunları atar, modelle üretilen bir "context satırı" ise
inference maliyetiyle geri satın alır. Yazarın zaten yazdığını kullan.

**Modele dokunmadan önce retrieval'ı doküman seviyesinde ölç.** Ucuz, deterministik bir kontrol —
doğru doküman ilk 1'e, ilk 5'e girdi mi — hatanın nerede olduğunu herhangi bir prompt
mühendisliğinden önce söyler. Bugün hiçbir metrik *cevabı* puanlamadı; o ikinci eval'ı da kur,
çünkü birincisi ikincinin bozuk olduğunu sana asla söylemeyecek.

<div class="measured">

Sadece eğitmen tarafı — lab reposunda `npm run eval`, 20 gold soru, `bge-m3`, doküman seviyesinde
hit@1 / recall@5 / MRR, 12 Eylül 2026'da tek koşu (sayılar başka makinelerde kayar):

| strateji | hit@1 | recall@5 | MRR |
|---|---|---|---|
| fixed-280 | 0.700 | 0.917 | 0.817 |
| fixed-280 + overlap 60 | 0.550 | 0.883 | 0.717 |
| recursive-600 | 0.700 | 0.900 | 0.806 |
| structure-1500 | 0.700 | 0.833 | 0.781 |
| structure-1500 + strip | 0.650 | 0.850 | 0.766 |

Dürüst oku: structure-aware chunking doğru **dokümanı** daha sık bulmuyor. **Chunk'ı** düzeltiyor —
K satırını header'ıyla birlikte — "bilmiyorum"u EUR 90'a çeviren şey bu. Overlap her metrikte
kaybediyor. Doküman seviyesindeki tek bir sayının neden yetmediği ve ne getirildiğini hâlâ neden
okuduğun da bu.

</div>

**Deployment birimi yeniden ingest'tir, yeniden training değil.** Q3 Q2'nin yerine geçtiğinde
pipeline değişmiyor, model değişmiyor, prompt değişmiyor. Yeni edisyonun tek bir ingest'i, içinde ne
olduğunu söyleyen tek bir collection adı — `kraken-2026-Q3-structure-1500` — ve bayat cevap gitti.
Bu, planlayabildiğin, geri alabildiğin, diff'leyebildiğin bir deploy. Fine-tune bunların hiçbiri
değil.

## Laptop'ında ne kalıyor

Bugün çalışan her şey hâlâ orada ve hiçbiri network istemiyor:

- `amadeus-rag-lab` clone'u — corpus, iki edisyon, `corpus/DELTA.md`, `src/pipeline.ts` içindeki
  TypeScript pipeline, ve eğitmenin `eval/` altındaki eval'ı;
- iki container — named volume'üyle `chroma` (son collection'ın hâlâ içinde) ve build edilmiş `api`
  imajı. `--build` olmadan `podman compose up` ikisini bıraktığın gibi geri getirir;
- Bruno collection'ı, `00-health`'ten `06-freshness`'a bütün klasörler, her request body'si
  bıraktığın gibi;
- host'taki iki Ollama modeli.

Pazartesi, network yokken: Ollama'yı başlat, **Terminal (`amadeus-rag-lab` repo kökü)** →
`podman compose up`, Bruno'yu aç, **Bruno — `00-health` › `health`** gönder. Cevap geliyorsa
collection'ın geri kalanı çalışır. `corpus/2026-Q3/` içinde bir dokümanı değiştir, yeniden ingest
et, yeniden sor — bütün döngü bu, ve artık senin.

## Günün bilerek dışarıda bıraktığı

Bilerek: hybrid retrieval ve BM25, reranking, contextual retrieval, agentic multi-hop döngüler. Her
biri bu kursun önceki bir sürümünde ölçüldü ve her birinin, lab'ın tek geçişli retrieval'ının henüz
karşılamadığı bir ön koşulu var. Eski sayılarla ve her birini denemek için `src/pipeline.ts`'e ne
ekleyeceğinle birlikte [Daha İleri](/tr/reference/going-further/) sayfasında yazılı. Tek dokümanlı
soru artık senin sorun olmaktan çıktığında oku.

<div class="presenter-note">
Gün 15:00'te kapanıyor ve bu blok onun son yirmi dakikası, ardından beş dakikalık tampon. EUR 90 cümlesiyle bitir, sonra çıkış
cümlesi, sonra konuşmayı kes — arkasına özet slaytı yok. Kesmeden önce tek bir el kaldırma:
"Pazartesi <code>podman compose up</code> çalıştıracak olan?" Say ve sayıyı salona geri söyle;
elindeki tek kalıcılık metriği bu. Repo linki ve daha-ileri sayfası toplanırlarken ekranda olsun,
öncesinde değil. Bu noktada Ollama çökmüşse önemi yok — burada hiçbir şey çalışmıyor.
</div>

## Çıkış cümlesi

> Bugünkü her adım, bir öncekinin yetmediği yerde doğdu. Sonuncusu yeniden training değil,
> yeniden ingest'ti — ve Pazartesi hâlâ çalışan sürüm de o.
