---
title: "6. Embedding Bake-Off"
description: "Varsayılan embedder senin dilin için doğru mu?"
---

## Gate sorusu

> **Varsayılan embedder senin dilin için doğru mu?**

Modül 5 tahtada açıklanmamış bir duvar bıraktı. 28 tam doküman üzerinde `bge-m3` hit@1 **0.600**
aldı; Türkçe soru / İngilizce doküman olan altı soruda ise **0.333** — altının dördü yanlış geldi.
İki şüpheli var: bu salonda kimsenin seçmediği embedder ya da dokümanları kesme biçimimiz. Bu modül
şüphelilerden birini eliyor.

Hiç seçmediğinle başla. Modelin `bge-m3` olduğunu söyledim ve şimdilik güvenmeni istedim. Kendi
projende ise `pip install chromadb` yazacaksın, `add()` çağıracaksın ve model senin yerine seçilmiş
olacak. O yüzden güvenmeyi bırak. Tek bir komut, aynı yirmi soru ve aynı chunk'lar üzerinde üç
embedder'ı puanlıyor ve puanı soru tipine göre ayırıyor:

```text
hit@1 by type         all-MiniLM-L6-v2  nomic-embed-text            bge-m3
--------------------------------------------------------------------------
tr_tr (n=4)                      0.750             0.500             1.000
tr_en (n=6)                      0.000             0.000             0.667   <-- Turkish question, English document
en_en (n=4)                      0.500             0.500             0.500
exact_token (n=4)                0.250             0.500             1.000
multi_hop (n=2)                  0.500             0.500             0.500
```

`tr_en` satırını oku. Bunlar Türkçe konuşan bir temsilcinin İngilizce dokümana soru sorduğu altı
soru — corpus'umuzun yarısının şekli. ChromaDB'nin kendi varsayılanı altıda sıfır alıyor. Makul
görünen, İngilizce öncelikli bir seçim olan `nomic-embed-text` de altıda sıfır alıyor. Exception
yok, uyarı yok, log satırı yok, şüphe uyandıran bir skor yok. İki pipeline da kendinden emin
görünen benzerlik skorlarıyla doküman döndürüyor ve doğru doküman aralarında değil. Generator da
kendisine verilen şeyden cevap üretiyor, çünkü işi bu.

**Tabloyu okumadan önce index koşulunu başlık satırından oku.** O koşu 154 structure-aware chunk
üzerinde — modül 7'nin splitter'ı, yani henüz elinde olmayan bir index. Bu bilerek böyle: embedder
ancak index çakılıysa tek değişken olabiliyor, o yüzden bake-off index'i bu kursun ulaştığı en iyi
hâline çakıyor. Bu iki yöne birden kesiyor ve iki yön de işe yarıyor. İngilizce öncelikli iki
modele kursun en iyi index'i verildi ve bir dil sınırı aşıldığında yine **0.000** aldılar; yani
hiçbir chunking onları kurtaramazdı ve embedder sorusu modül 7'yi beklemeden burada kapanıyor. Bir
de 0.750 senin sayın değil. Seninki hâlâ tam dokümanlar üzerinde 0.600 ve `tr_en`'in hâlâ 0.333.

<div class="presenter-note">
Slot: 11:26'da 16 dakika. Bu modül kesme listesinin başında, 8 dakikaya iniyor — kesme planı bu
sayfanın son notunda.<br />
Hiçbir şey çalıştırmadan önce modül 5'in 0.333'ünü tahtaya geri yaz; bu modülün bir sayıyı
kımıldatabilmesi için salonun elinde bir sayı olması gerekiyor. Sonra taahhüt al: "sadece embedding
modelini değiştiriyoruz, başka hiçbir şeyi. Altı Türkçe sorudan kaçında doğru dokümanı hâlâ
getirir? Dört ve üzeri diyenler el kaldırsın." Ellerin çoğu kalkar. <code>tr_en</code> satırını
göster ve üç saniye hiçbir şey söyleme. O sessizlik sonraki paragraftan daha çok iş yapar.
4 dakika.
</div>

## Embedding gerçekte nedir

Bir embedding modeli bir string alır ve sabit uzunlukta bir float listesi döndürür. Sözleşmenin
tamamı bu. `bge-m3` her girdi için **1024** sayı döndürür — tek kelime de versen bir sayfa da.
`nomic-embed-text` **768**, `all-MiniLM-L6-v2` **384** döndürür. Ama *girdi* sınırsız değil:
MiniLM 256 token'da okumayı bırakıyor, yani 900 karakterlik her chunk'ın bir kısmı ona hiç
ulaşmıyor. Varsayılanın sana söylemediği bir şey daha.

Her sayı bir koordinat ve o noktaların ikisiyle yaptığın tek şey ne kadar yakın olduklarını sormak.
Yakınlık cosine similarity ve `eval/retrieval.py` içinde dört satır:

```python
def cosine(a, b):
    dot  = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / (norm + 1e-12)
```

Bütün günün dayandığı iddia şu: o uzaydaki yön anlama karşılık geliyor. Modelin eğitim dağılımının
içinde, yaklaşık olarak geliyor. Dışında ise sayılar gelmeye devam ediyor ve sessizce varsaydığın
şeyi ifade etmeyi bırakıyor.

## Tuzak, canlı

0.000 skoru sana doğru dokümanın birinci olmadığını söylüyor. Onun yerine ne geldiği asıl işe
yarayan yarısı ve `nomic-embed-text` neredeyse her seferinde aynı cevabı veriyor:

```text
q05 ['sop_misconnect_v4']              -> macro_tr_noshow
q06 ['sop_denied_boarding']            -> macro_tr_noshow
q07 ['interline_xx_yy']                -> macro_tr_noshow
q08 ['codeshare_xx_yy_conditions']     -> macro_tr_noshow
q09 ['macro_en_refund']                -> macro_tr_rebook
q10 ['macro_en_special_assistance']    -> macro_tr_noshow
```

Altının beşi aynı Türkçe dokümana düşüyor. `macro_tr_noshow` makul bir komşu — Türkçe, cezaları
anlatıyor, içinde "iptal cezası iki katına çıkar" cümlesi geçiyor — ama tutar vermeyi açıkça
reddediyor: *"tutarlar bu makroda tekrarlanmaz"*, İngilizce ücret kuralına bak diyor. Retriever
doğru dili seçti ve cevabı içeren her dokümanı eledi.

## Neyin bozulduğunu tam söyle

Kolay özet "bu modeller Türkçe bilmiyor" demek. Bu özet yanlış ve soru-cevapta seni yakar. `tr_tr`
satırına geri dön: Türkçe sorulara karşı Türkçe dokümanlarda `all-MiniLM-L6-v2` **0.750**,
`nomic-embed-text` **0.500** alıyor — zayıf, ama sıfırın yanından bile geçmiyor. `tr_en`'de ikisi de
**0.000** alıyor. Altıda altı. Bozulan şey Türkçe değil. Diller **arasında geçiş**.

Tek dilli bir modelin uzayında metnin dili başlı başına güçlü bir yön: alakasız konulardaki iki
Türkçe cümle, bir Türkçe cümle ile onun kendi İngilizce çevirisinden daha yakın durabiliyor, çünkü
"Türkçe olmak" "iptal cezasıyla ilgili olmak"tan ağır basıyor. Dil, anlamdan büyük bir eksen hâline
geliyor ve yukarıdaki `tr_en` listesi tam olarak bunun içeriden görünüşü.

Doğal olarak bakacağın hiçbir metrikte görünmüyor — TR→TR bir şekilde çalışıyor, EN→EN çalışıyor ve
`en_en`'de üç model de 0.500'de berabere. Corpus'un yarısı, soruların yarısı, sessizce bozuk;
üstelik yalnızca İngilizce bir test seti bunu asla bulamazdı. Kırılmaya yol açan model de
ChromaDB'nin varsayılan embedding function'ı `all-MiniLM-L6-v2`; ilk `add()` çağrında sen istemeden
iniyor. Modül 8 onun senin yerine seçtiğini gösteriyor ve faturayı da gösteriyor.

<div class="presenter-note">
Ağzında gevelenmemesi gereken cümle: <strong>bu modeller Türkçede çaresiz değil — yapamadıkları şey
diller arası eşleşme ve senin corpus'un karışık.</strong> Bir kez, yavaş söyle ve tahtaya
"TR → TR: 0.750 / TR → EN: 0.000" yaz. Ollama burada düşerse salonun önünde debug yapma:
<code>eval/RESULTS.md</code> bölüm 3 bu sayfanın ilk tablosu, bölüm 7 ise çalıştırmak üzere olduğun
tip kırılımı. <code>tr_en</code> satırını bölüm 3'ten oku — <code>bge-m3</code> için 0.667, diğer
ikisi için 0.000 — ve bunu geçen hafta bir laptopta o komutun ürettiğini söyle. 3 dakika.
</div>

## Ne çalıştırıyorsun

**Bu sefer notebook yok.** Bu modül tek bir komut.

**Terminal (repo kökü):**

```bash
python exercises/m6_embedding_bakeoff.py
```

- **ne görmen gerekiyor** — önce `28 documents, 154 structure-aware chunks, 20 questions` başlık satırı, sonra üç embedder'ın her birinin index'leyip sorguladığını bildirmesi, sonra iki tablo. Önemli satır `tr_en (n=6)`: `0.000  0.000  0.667` yazıyor ve yanında onu gösteren bir ok var. Hiçbir şey hata vermiyor, hiçbir şey uyarmıyor
- **kabaca ne kadar sürüyor** — yaklaşık 18 saniye; çoğu, 154 chunk'ı üç kez embed etmek

Burada hiçbir şey indirilmiyor: `bge-m3` ve `nomic-embed-text` evde çekildi, `all-MiniLM-L6-v2` ise
`scripts/seed_offline_assets.py` tarafından cache'lendi ve `scripts/verify_setup.py` bunu kontrol
ediyor. MiniLM sütunu `SKIPPED` diyorsa aynı terminalde `python scripts/seed_offline_assets.py`
çalıştır ve bake-off'u tekrarla — bu salonda bitmeyecekse de devam et, çünkü `tr_en` satırı iki
embedder'la da 0.000'a karşı 0.667 diyor.

<div class="presenter-note">
Projeksiyonda çalıştır ve salon da aynı anda çalıştırsın; yaklaşık 18 saniyede kimse beklemiyor.
İki şeye dikkat et. <code>all-MiniLM-L6-v2</code> sütununda <code>SKIPPED</code> yazan bir laptop
seed script'ini hiç çalıştırmamıştır — iki embedder'lık sonucu al ve devam et. Bir de biri satırları
tarayıp ucuz modellere savunma arayacak. İki tane var ve ikisinin de cevabı belli. Üç model
<code>en_en</code>'de 0.500'de berabere: doğru, ve zaten mesele bu — İngilizce sorularda ucuz
embedder gözle görülür biçimde kötü değil, bu yüzden kimse yakalamıyor. Üç sütun da
<code>multi_hop</code>'ta 0.500'de berabere, <code>bge-m3</code> dahil — bu, karşılaştırmanın en
iyi ve en kötü modelinde aynı, iki sorudan biri; yani embedder'ı değil, chunking'in o soruyu
ulaşılabilir kılmasını ölçüyor. Tartışmaya dönüşmeden söyle. 5 dakika.
</div>

## Sayılar ne dedi

<div class="measured">

Üç embedder, 20 soru, structure-aware chunk'lar:

| embedder | vektör uzunluğu | hit@1 | recall@5 | MRR | `tr_en` (n=6) |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` — ChromaDB'nin sessiz varsayılanı | 384 | 0.350 | 0.600 | 0.467 | 0.000 |
| `nomic-embed-text` | 768 | 0.350 | 0.633 | 0.490 | 0.000 |
| `bge-m3` | 1024 | **0.750** | **0.833** | **0.817** | **0.667** |

Aynı model, iki index — ya koşulu da söyle ya hiç söyleme:

| `bge-m3` şunun üzerinde | hit@1 | `tr_en` (n=6) |
|---|---|---|
| tam dokümanlar — şu anda elinde olan pipeline | 0.600 | 0.333 |
| structure-aware chunk — modül 7'nin bittiği yer | 0.750 | 0.667 |

</div>

MiniLM'in MRR'ına bir dipnot: iki Ollama modeli 154 chunk'ın tamamı üzerinde sıralanıyor, ama
Chroma'dan ilk 20 isteniyor; o pencerenin dışında kalan doğru chunk gerçek reciprocal rank'i yerine
0 alıyor. hit@1, recall@5 ve `tr_en` sütununun tamamı bundan etkilenmiyor; yalnızca o MRR modelin
hak ettiğinden sert okunuyor.

Corpus: 28 doküman, 154 structure-aware chunk; gold set 20 soru, 6'sı Türkçe sorgu / İngilizce
doküman ve her şey lokal Ollama üzerinde. Yirmi soru iki tasarım arasında seçim yapmaya yeter,
yayımlamaya fazlasıyla azdır; yani 0.05'in altındaki bir fark gürültüdür. Bu ikisi değil: 0.350'ye
karşı 0.750 sekiz soru ve `tr_en` sütunu dörde karşı sıfır.

Yani embedder chunking'den önce kapanıyor ve nedeni ikinci tablo. Daha iyi bir index, diller arasını
görebilen modele 0.333 → 0.667 kazandırdı. Diğer ikisine hiçbir şey kazandırmadı, çünkü onlar
0.000'ı alırken zaten modül 7'nin index'inin üstünde duruyorlardı.

## Daha derine

Multilingual eğitimin değiştirdiği şey mimari değil, objective. Tek dilli bir model tek dil içinde
aynı anlama gelen metinleri birbirine çekmek üzere eğitilir. `bge-m3` ise yaklaşık yüz dilde paralel
ve mined çiftlerle eğitilir; pozitif çift bir cümle ve onun çevirisidir — dolayısıyla Gradient'in
tek işi Türkçe cümleyle İngilizce cümleyi aynı noktaya indirmek ve dil, loss onu cezalandırdığı için
kullanılabilir bir yön olmaktan çıkıyor. Daha çok veri değil, daha iyi prompt da değil: sen
weight'leri çekmeden aylar önce pişmiş.

Boyut sayısı kalite değil. Burada 1024, 768'i ve 384'ü yeniyor ve bu sezgiyi okşuyor; ama MiniLM'i
bozan şey genişlik değil — İngilizce soru / İngilizce dokümanda hâlâ 0.500 alıyor ve tam olarak
0.000'ı yalnızca bir dil sınırı aşıldığında veriyor. Genişlik bir maliyet ve maliyet gerçek: eşit
koşulda, Ollama üzerinden, `bge-m3` aynı 154 chunk'ı index'leyip sorgulamak için **18.5 s**,
`nomic-embed-text` ise **16.0 s** harcadı.

Kendi corpus'un için embedder değerlendirmek bir öğleden sonra sürüyor ve bu konudaki en değerli
öğleden sonra. Kullanıcılarının gerçekten yazdığı dilde yirmi soru yaz, her birini cevabı içeren
dokümanla etiketle. Şüphelendiğin hataya göre grupla: aynı dil, diller arası, tam tanımlayıcı,
paraphrase. Sonra üç dört modeli `DenseRetriever`'dan geçir ve hit@1'i ortalama olarak değil **grup
grup** yazdır. İşi gruplama yapıyor: MiniLM'in genel 0.350'si kötü ama alarm verici değil, temiz
sıfırı sadece `tr_en` satırı gösteriyor.

0.667 de bir zafer değil. Altıda ikisi hâlâ yanlış. Bake-off'un kazananı ölçtüğün en az kötü
seçenektir, çözülmüş bir problem değil; 0.667'de hâlâ bozuk olan şey de modül 7'nin konusu.

<div class="presenter-note">
Biri mutlaka "sorguyu önce İngilizceye çevirsek olmaz mı?" diye soracak. Ciddiye al — multilingual
embedder'lar iyi olmadan önce insanlar tam da bunu yapıyordu. Cevabı maliyetle ver: sorgu başına
fazladan bir model çağrısı, çeviri CLASSIC K gibi bir terimi düşürdüğünde yeni bir hata modu ve
sonunda yine bir embedder seçmen gerekiyor. Ölçülebilir olduğunu ve bizim ölçmediğimizi söyle.
Sayı uydurma. Kapanış hesabıyla birlikte 4 dakika, sonra devam.<br />
<strong>16 dakikayı 8'e indirmek.</strong> Modül 5'in 0.333'ünü tahtada tut ve <code>tr_en</code>
satırını göster (4 dk), diller arası geçişle ilgili tek cümleyi (1 dk), koşuyu ve ilk tabloyu
(2 dk), çıkış cümlesini (1 dk). <em>Tuzak, canlı</em>, <em>Daha derine</em> ve çeviri sorusunu at.
Index koşulu paragrafını atma: modül 7, 0.600 ile açılıyor ve salonun bunun neden 0.750 olmadığını
bilmesi gerekiyor.
</div>

## Çıkış cümlesi

> Varsayılan senin dilin için yanlıştı ve kimse sana söylemedi — üstelik zaten elinde olan embedder
> modül 5'i bozan şey değil. Tam dokümanlar üzerinde `tr_en`'in hâlâ 0.333 ve geriye tek bir şüpheli
> kaldı: kesme.
