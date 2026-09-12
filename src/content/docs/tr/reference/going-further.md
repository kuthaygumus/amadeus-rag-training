---
title: "Daha İleri: Hybrid, Rerank, Agentic"
description: "Her RAG yazısının sana bir sonraki adım diye uzattığı dört teknik — ne oldukları, bu kursun önceki sürümünün ne ölçtüğü, ve her birini denemek için lab'a ne ekleyeceğin."
---

> **Günün ötesi — 7 Ekim'de anlatılmıyor, lab'da yok.** Bu sayfadaki hiçbir şey Bruno
> collection'ından çalışmıyor. Sonraki haftanın okuması: tek dokümanlı soru artık senin sorun
> olmaktan çıktığında.

Lab bilerek seçilmiş bir noktada duruyor: tek soru, tek doküman, tek geçişlik retrieval ve bulunan
doğru chunk. "RAG'i iyileştir" diye arattığında karşına ilk çıkacak şey dört kelime — hybrid, rerank,
contextual, agentic. Bu kursun Python notebook'ları üzerine kurulu önceki sürümü, bunların üçünü aynı
yirmi gold soruya karşı, aynı 28 Kraken Air / Wyvern Overseas dokümanı üzerinde çalıştırdı ve ölçtü.
Dördünden birini gün içinde fark etmeden zaten yaptın. İkisi kaybetti. Dördüncüsü lab'ın sana hiç
göstermediği bir problemi çözüyor ve bedeli neyse onu ödetiyor.

Eski sayılar burada tek bir sebeple, sadece kutuların içinde duruyor: her tekniğin ihtiyaç duyduğu
**ön koşulu** adlandırıyorlar ve senin sistemine taşınan kısım o. Rakamlar taşınmıyor.

<div class="measured">

**Bu sayfadaki her sayı hakkında.** Bu kursun önceki Python sürümünden geliyorlar (Eylül 2026,
`eval/RESULTS.md`), şimdi reponun git geçmişinde — `amadeus-rag-training` deposunun `07c13ba`
commit'i. Farklı bir chat modeliyle, farklı bir chunker konfigürasyonuyla (lab'ın 132'si değil, 154
structure-aware chunk) ve lab'da artık bulunmayan puanlama koduyla üretildiler. **`amadeus-rag-lab`
ile yeniden üretilemezler**; rakamları için değil, yönleri için alıntılanıyorlar. Yirmi soru iki
tasarım arasında karar verir; 0.05 civarının altındaki bir fark gürültüdür.

</div>

## BM25 ve paraphrase'de neden çöküyor

BM25 keyword retriever'ı: bir terim corpus'ta nadirse yüksek puan alır, aynı dokümanda tekrar ettikçe
ağırlığı doyar, uzun dokümanlar cezalandırılır. `XX 1487` gibi bir tanımlayıcıyı anında bulur ve
hiç model istemez — embedding çağrısı yok, GPU yok, mikrosaniyelerde güncellenen bir inverted index.

Tam olarak tek bir yerde çöker ve corpus o yeri yüzüne vurur: **sorgu ile dokümanın kelime hazinesi
ortak olmalı.** "iptal edersem ne öderim"den "Cancellation penalty"ye giden yol yok. Aynı dildeki bir
paraphrase aynı hatanın hafif hâli — "bu bileti bırakmak bana neye patlar", RULE 2A tablosuyla tek
bir içerik kelimesi paylaşmıyor. Embedder iki boşluğu da aşıyor çünkü token'ı değil anlamı
haritalıyor; BM25 aşamıyor, chunking de durumu kötüleştiriyor: eskiden altı terimin altısını tek
dokümanda bulan altı terimli sorgu, şimdi onları üç chunk'a dağılmış buluyor ve hiçbiri fazla puan
almıyor.

<div class="measured">

Önceki Python kursu, Eylül 2026, 20 gold soru; 6'sı Türkçe soru / İngilizce doküman (`tr_en`),
4'ü Türkçe / Türkçe (`tr_tr`):

| aynı 154 structure-aware chunk üzerinde retriever | hit@1 | recall@5 | MRR | `tr_en` |
|---|---|---|---|---|
| dense, `bge-m3` | **0.750** | **0.833** | **0.817** | 0.667 |
| BM25 | 0.300 | 0.633 | 0.467 | **0.000** |

Bütün dokümanlar üzerinde, herhangi bir chunking'den önce, BM25 0.400 hit@1'e ulaştı — ve dört
`tr_tr` soruda **1.000**, altı `tr_en` soruda **0.000**. Aynı corpus, en iyisi ve en kötüsü,
dokümanın hangi dilde olduğuna göre. Geri kalanını chunking aldı: exact-identifier skoru 0.750'den
0.500'e düştü.

</div>

**Lab'da denemek için:** `ingest()`'in Chroma'ya yazdığı aynı chunk metinleri üzerinde bir BM25
index'i kur ve sıralamasını `src/pipeline.ts` içindeki `retrieve()`'in dense sıralamasının yanında,
aynı `hits` biçiminde döndür ki ikisi yan yana kıyaslanabilsin.

## Hybrid retrieval ve reciprocal rank fusion — fusion neden seyreltti

Hybrid, iki retriever'ı da çalıştırıp birleştirmek demek. Herkesin kullandığı birleştirme
reciprocal rank fusion: cosine similarity ile BM25 skorunun ortak birimi yok, o yüzden skorları atıp
sıraları tutuyorsun. Her doküman, göründüğü her sıralamadan `1 / (k + sıra)` kazanıyor, `k = 60`
gelenek, toplamlar sıralanıyor. Gerçekten zarif — ve kimsenin yüksek sesle söylemediği bir ön koşulu
var.

`k = 60`'ın seni koruduğunu varsaymadan önce aritmetiği yap. 1. sıra `1/61` ediyor; 3. sıra neredeyse
aynı; 154. sıra hâlâ **1. sıranın %28'i**. İki retriever de *aynı* chunk kümesinin tamamını
sıralıyorsa — ki öyle, "sadece biri buldu" durumu yok — fusion iki yumuşatılmış reciprocal rank'in
toplamına dönüşür ve zayıf sıralama her şeye, en dibe kadar oy verir. **Reciprocal rank fusion iki
girdinin de bağımsız olarak sağlam olduğunu varsayar.** İyi bir retriever ile corpus'un üçte birinde
sistematik olarak yanılan biri ortada bir yerde buluşmaz; hatayı miras alırlar, çünkü bir sıralama
her zaman bir fikir gibi görünür.

<div class="measured">

Önceki Python kursu, Eylül 2026:

| aynı 154 chunk üzerinde retriever | hit@1 | recall@5 | MRR | `tr_en` |
|---|---|---|---|---|
| dense, `bge-m3` | **0.750** | **0.833** | **0.817** | 0.667 |
| BM25 | 0.300 | 0.633 | 0.467 | 0.000 |
| ikisinin RRF'i | 0.450 | 0.683 | 0.579 | **0.000** |

Dense tek başına 0.750; ikinci retriever'ı eklemek 0.300'e — altı soruya — patladı. `tr_en`'de 0.667
ile 0.000, 0.000'a fuse oldu. Fusion tam bir kez kazandı: BM25'in eşleşecek bütün dokümanlara sahip
olduğu **doküman** seviyesinde RRF sayfanın en iyi recall@5'ini aldı — dense'in 0.717'sine karşı
0.733 — ve o 0.016'yı 0.100 hit@1 ile ödedi.

</div>

Hybrid'in karşılığını verdiği yer: milyonlarca dokümanda BM25, pahalı bir ikinci aşamaya birkaç yüz
aday uzatan ucuz bir **ilk aşama** olarak geri döner. Orada cevap üstüne oy vermek yerine aday
üretir ve metriği recall@k'dır. Yukarıda ölçülen hybrid o değil.

**Lab'da denemek için:** iki sıralamayı `retrieve()` içinde, threshold filtresinden önce konumlarına
göre fuse et ve asıl taradığın düğme, zayıf sıralamanın ne kadar derine oy vermesine izin verdiğin
olsun.

## Reranking — listwise, pointwise ve reranker ne zaman karşılığını verir

Reranker en üstteki adayları alır ve bir modele değerlendirtir. En ucuzu prompt'lu bir chat modeli
ve iki biçimi var. **Listwise**: tek çağrı, bütün adaylar, "bunları sırala". **Pointwise**: aday
başına bir çağrı, "bu pasajı bu soru için 0–10 puanla"; eşitlikte retriever'ın sırası korunur, yani
reranker bir dokümanı ancak gerçekten bir fikri varsa oynatır.

Önceki kurs iki çağrıyı da aynı soruda, aynı altı aday üzerinde yaptı. Pointwise altı çağrıda altı
kullanılabilir tam sayı döndürdü. Listwise `1,4,2,5` döndürdü — altı pasaj için dört indeks,
sıralayacak bir şey yok. **Küçük bir model dar soruyu iyi, geniş soruyu kötü cevaplar.** Pointwise
ayrıca cevap okunamazsa hiçbir şey yapmamaya geriler; listwise'ın böyle bir tabanı yok. Pointwise
altı çağrıya, listwise bire patlar.

Sonra rerank yapılır mı yapılmaz mı sorusunu karara bağlayan sayı. Aynı pointwise reranker, kalitesi
bilerek farklı tutulmuş dört retrieval kurulumunda çalıştırıldı:

<div class="measured">

Önceki Python kursu, Eylül 2026 — reranker, ilk 8'i puanlayan 3B'lik bir chat modeli, 20 soru:

| retrieval kurulumu | hit@1 önce | sonra | MRR önce | sonra | sonuç |
|---|---|---|---|---|---|
| zayıf embedder + fixed-280 | 0.350 | **0.450** | 0.515 | **0.544** | yardım etti |
| zayıf embedder + structure-aware | 0.350 | **0.400** | 0.490 | **0.540** | yardım etti |
| `bge-m3` + fixed-280 | 0.700 | 0.500 | 0.817 | 0.680 | zarar verdi |
| `bge-m3` + structure-aware | **0.750** | 0.600 | 0.817 | 0.725 | zarar verdi |

Soru soru, en güçlü kurulumda: 20 sorunun 15'inde gold doküman zaten 1. sıradaydı ve reranking
bunların **4**'ünü aşağı itti; olmayan 5'in **1**'ini yukarı çekti. Maliyet: soru başına 8 model
çağrısı, kurulum başına 160, tablo için 640; dört kurulum tek bir M-serisi Mac'te 191 s, 127 s, 51 s
ve 71 s sürdü.

</div>

İki zayıf kurulumu yukarı çekti, iki güçlüyü aşağı indirdi; istisna yok. Tabloyu bölen chunking
değildi — iki strateji de iki tarafta var — embedder'dı. **Reranker düzler, ve kendi tavanına doğru
düzler.** Önce kurulumlar 0.350'den 0.750'ye yayılıyordu; sonra 0.400'den 0.600'e. O kalitede bir
retriever'dan aday aldığında o reranker'ın çıktısı, ne verilirse verilsin o banda indi. Retriever'ın
bu eşleşmeden kötüyse modelin fikrini dayatmak bir upgrade. Zaten iyiyse dayatmak ancak kaybettirir —
ve "yardım ettiği" iki satır bile yanlış tasarım olurdu, çünkü oradaki doğru hamle embedder'ı
düzeltmekti.

Production reranker'ı chat modeli değil, **cross-encoder**: sorgu ile pasaj tek forward pass'te
birlikte okunur, relevance etiketleriyle eğitilmiştir. `bge-reranker-v2-m3` çok dilli olanı ve
lab'ın embedder'ıyla eşleşiyor. Önceki kurs onu ölçmedi; "0.750'yi geçerdi" bir hipotez ve ders iki
cevapta da ayakta kalıyor, çünkü daha güçlü bir hakemin de retriever'ına göre konumlandırman gereken
bir tavanı var.

**Lab'da denemek için:** `/query` route'unda `retrieve()` ile prompt'un kurulduğu yer arasına, hit
başına 0–10 arası bir tam sayı döndüren birer `gemma3:4b` çağrısı ekle, hit'leri eşitlikte
retriever'ın sırası korunarak buna göre yeniden sırala ve az önce eklediğin çağrıları say.

## Contextual retrieval — onu zaten yaptın

Contextual retrieval, her chunk'ın başına onu tek başına anlaşılır kılacak kadar çevre bilgisi koymak
demek; genelde bir modele "bu chunk dokümanın neresinde" diye bir satır yazdırarak — ingest anında,
chunk başına bir generation çağrısıyla.

Lab'ın `structure` stratejisi her chunk'ın başına başlık yolunu zaten koyuyor:
`[fare classic shorthaul > RULE 2A …]` — doküman başlığı artı chunk'ın altında durduğu başlık. Aynı
mekanizma, modelle üretilmiş değil dokümandan alınmış, inference maliyeti sıfır. K satırı ile sütun
başlığının aynı chunk'ta hayatta kalmasının ve modül 7'nin 12 Eylül koşusunda "EUR 13"ü EUR 90'a
çevirmesinin sebebi bu. Başlıkları olan dokümanlarda, üretilmiş bir context satırının Markdown'ın
sana bedavaya vermediği bir şey satın aldığından emin ol; olmayanlarda — taranmış PDF'ler, chat
logları, düz metin — denenecek ilk şey bu.

**Lab'da denemek için:** `ingest()` içinde, embedding'den önce, chunk başına chat modelini bir kez
çağırıp tek satırlık bir "burası nerede" prefix'i al ve sonucu `structure`'ın yanına yeni bir
strateji olarak ingest et ki `npm run eval` paralı prefix'i bedavasıyla kıyaslayabilsin.

## Agentic RAG — döngü, multi-hop ve fatura

Gün içindeki her kaldıraç tek bir geçişe etki ediyor: soruyu bir kez embed et, bir komşuluk geri
gelsin. Daha iyi embedder noktayı oynatır; daha iyi chunker her adayı netleştirir; reranker geleni
yeniden sıralar. Hiçbiri ikinci bir sorgu eklemez. Bazı sorular ister.

Önceki kursun gold set'inde böyle iki soru vardı. Birini, bir çağrı merkezi görüşmesindeki hâliyle:
Kraken uçuşu gecikti, CLASSIC K sınıfındaki yolcu CDG'de Wyvern bağlantısını kaçırıyor ve beş saat
bekleyecek — beklemede ne verilmeli, Wyvern bacağı yeniden fiyatlandırılır mı, yolcu kendi isteğiyle
yarına geçmek isterse değişiklik ücreti ne? Üç soru tek bir paltoda. Bekleme sırasındaki bakım
`sop_misconnect_v4.md`'de; Wyvern kuponunun hayatta kalıp kalmadığı `interline_xx_yy.md`'nin 4.
maddesinde; değişiklik ücreti `fare_classic_shorthaul.md`'de. Hiçbir dosya üçünün ikisini
tutmuyor, bilerek: SOP hiçbir türde parasal değer belirlemiyor, interline maddesi hiçbir şeyi
fiyatlamıyor. Dokümanlar birbirini gösteriyor. Gerçek kural kitapları da öyle.

### Hangi fare sheet — RULE 7

Bir bileti iki fare sheet yönetebilir ve corpus bunu ikisinde de geçen bir kuralla karara bağlıyor —
`fare_classic_shorthaul` RULE 7 ve `fare_classic_longhaul` RULE 7: **sheet'i doküman değil,
işlem seçer.** Tek bir kupondaki gönüllü değişiklik, elde tutulan Kraken sektörünün bandının
sheet'inde değerlendirilir; yolculuğun tamamının iptali veya iadesi long-haul sheet'te. Yani
misconnect sorusunun değişiklik ücreti short-haul CLASSIC K: **EUR 70**. Aynı yolculuğun iptali
long-haul CLASSIC K olurdu: **EUR 195**. Günün sayısı EUR 90'ı — short-haul iptali — söylemek,
yanlış sheet'te doğru satır olurdu. Tek bir geçiş, iki dokümanın da sonuna gömülü bir kuralı fark
edemez; değişiklik mi iptal mi fiyatladığını bilen bir tool ise sheet'i retrieval'dan *önce* seçer.

### Döngü

Dört adım, framework yok. **Decompose** — tek model çağrısı soruyu en fazla dört tane bağımsız alt
soruya çeviriyor. **Alt soru başına retrieve** — her biri kendi embedding'ini ve aynı collection'a
karşı kendi top-k'sını alıyor: bir nokta yerine üç. **Yeterlilik kontrolü** — tek model çağrısı `YES`
diyor, ya da `NO` artı eksik olan için tek kısa sorgu; dar, pointwise'ın listwise'ı yendiği şekilde.
**Kaynaklı cevap**, her sayının arkasındaki dokümanın adıyla. Retrieval modelden önceki bir adım
olmaktan çıkıp modelin çağırdığı bir tool'a dönüşüyor — ne zaman duracağına ve sonra ne soracağına o
karar veriyor; bununla ne yapılacağına senin kodun.

<div class="measured">

Önceki Python kursu, 9 Eylül 2026'da kaydedilmiş koşu, tek bir M-serisi Mac:

| | ölçülen |
|---|---|
| gold set'teki `multi_hop` soru | 20'de 2 |
| bunların hit@1'i, tek geçiş, her chunking stratejisi | 0.500 — chunking'den önce 0.000 |
| misconnect sorusu: ulaşılan gold doküman, tek geçiş → döngü | **3'te 0 → 3'te 2**, 4.9 s |
| yolculuğun tamamı iptal sorusu: tek geçiş → döngü | 3'te 2 → 3'te 2, 7.9 s |
| soru başına model çağrısı, döngü | 6–10, üçü dördü embedding |

Döngü `interline_xx_yy`'yi hiç bulmadı. Buna rağmen yazdığı cevap yemek kuponunu ve EUR 70
değişiklik ücretini doğru verdi, otel eşiğini ve yeniden fiyatlandırma sorusunu sessizce düşürdü ve
kimsenin sormadığı bir iptal cezası ekledi. İki soru bir yöntemi ölçemez; bir yöntemin var olduğunu
ve bedava olmadığını gösterir.

</div>

### Fatura

**Daha çok çağrı** — basit RAG'in bir çağrı yaptığı yerde altı ila on, ve döngü çağrı başı
gecikmeyi tur sayısıyla toplamaz, çarpar. **Non-determinism** — plan üretiliyor, yani iki koşu
farklı decompose edebilir, farklı getirebilir, farklı kaynak gösterebilir; her alt soruyu logla,
debug edeceğin liste o. **Daha çok geçiş, yanlış şeyi çekmek için daha çok fırsat** — corpus'ta
yürürlükten kalkmış `sop_misconnect_v3.md` var: v4'ün EUR 15 ve 6 saatine karşı EUR 10 ve 8 saatlik
otel eşiği; chunk'lar modele ulaşmadan önce sürüme göre filtrele — sıralamaya dair bir umut olarak
değil, tool'un bir özelliği olarak. **Hiç bitmeyen döngü** — yeterlilik kontrolü her zaman "hâlâ bir
şey eksik" diyebiliyorsa der. Dört yerden sınırla: sert bir tur sınırı, alt soru sayısına sınır, bir
tur yeni bir şey getirmiyorsa dur, ve son turda elindekiyle cevap verip bulamadığını *söyle*. Dürüst
bir kısmi cevap sonsuz döngüden de, modül 1'de çıplak modelin ürettiği kendinden emin uydurmadan da
iyidir.

Ve bütün kursun taşıdığı çekince: yukarıdaki her metrik **retrieval**'ı puanlıyor — doğru doküman
geldi mi. Hiçbiri üstüne kurulan cümlenin doğru olup olmadığını puanlamıyor. Kaydedilmiş döngü koşusu,
ikisinin ayrışabileceğinin kanıtı.

**Lab'da denemek için:** `retrieve()`'i döngüde çağıran bir `/agent` route'u ekle — decompose
prompt'u, alt soru başına retrieval, dar bir yeterlilik prompt'u — sert bir tur sınırıyla ve üretilen
her alt sorunun `debug` alanına loglanmasıyla; önüne de bir router koy ki tek adımlı sorular
decomposition'ın bedelini hiç ödemesin.

## Eski sayıları nereden okursun

Tam tablolar — tür kırılımıyla chunking merdiveni, üç embedder, iki granülaritede BM25 ve fusion, dört
kurulumlu rerank taraması — `amadeus-rag-training` deposunun git geçmişinde, `07c13ba` commit'indeki
`eval/RESULTS.md`'de; onları üreten notebook'larla birlikte. Onları önceki kursun, önceki kursun
kurulumunu ölçmesi olarak oku. Lab'ın kendi sayıları eğitmen tarafındaki `npm run eval`,
`amadeus-rag-lab/eval/results.md` içinde — laptop'ındakini anlatan sayılar yalnızca onlar.
