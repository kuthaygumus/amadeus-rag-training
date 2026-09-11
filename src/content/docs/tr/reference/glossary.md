---
title: "Sözlük"
description: "Günde geçen her terim, bir kez tanımlanmış, ölçtüğümüz yerde sayısıyla birlikte."
---

Terimler, onları ilk kullanan modüle göre gruplanmıştır. Bir şeyi ölçtüysek sayısı da burada —
kontrol edemediğin tanımı yanlış hatırlarsın. Bu sayfadaki her sayı `eval/RESULTS.md`'den geliyor.

## Model ve eğitim

**Ağırlık (parametre)** — modelin yapıldığı sayılar; modül 2'deki rakam sınıflandırıcısında
101 770 tane var. Eğitim bittikten sonra bir daha değişmiyorlar; fine-tune edilmiş bir modelin
bayatlamasının sebebi bu.

**Training (eğitim)** — ağırlıkları ayarlayan döngü: girdiyi ileri çalıştır, ne kadar yanlış
olduğunu ölç, her ağırlığı işe yarayan yönde biraz oynat, tekrarla. Modül 2 bunu 60 000
görüntüyle beş kez, 0.92 saniyede yapıyor.

**Loss** — modelin şu anda ne kadar yanlış olduğunu söyleyen tek sayı. Eğitim, onu küçültme
sürecinin adı. Bizimki ilk batch'te 2.35'ten son batch'te 0.03'e düşüyor.

**Epoch** — eğitim verisinin tamamı üzerinden bir geçiş. Öğrenmenin çoğu ilkinde oluyor:
doğruluk %9.9'dan %95.35'e fırlıyor, kalan dört epoch'ta %97.47'ye sürünüyor ve tepesi epoch 4.

**Fine-tuning** — halihazırda eğitilmiş bir modeli kendi verinle daha ileri eğitmek. Çalışıyor
ama sonuç yine donmuş oluyor: modül 3'ün adapter'ı 2026-Q2 kitabının EUR 120'lik CLASSIC K
cezasını öğreniyor ve 2026-Q3'ün aynı satırı EUR 90 fiyatladığını fark edemiyor.

**LoRA** — Low-Rank Adaptation. Büyük ağırlık matrisini dondur ve çarpımı ona eklenen çok daha
küçük iki matrisi — *adapter*'ı — öğren. **Rank** kapasite, **alpha** ölçek. Modül 3 modelin
parametrelerinin %2.34'ünü eğitiyor; sonucun birkaç megabyte olmasının sebebi bu.

**QLoRA** — dondurulmuş base modelin 4 bit'e quantize edildiği LoRA. Base sadece okunuyor, hiç
yazılmıyor; bu yüzden hassasiyet kaybı beklediğinden ucuza geliyor ve 7B'lik bir model tüketici
donanımında eğitilebilir hâle geliyor.

**RLHF / DPO** — cevaplar üzerinden değil karşılaştırmalar üzerinden eğitim: iki cevap ve insanın
hangisini tercih ettiği. RLHF ayrı bir ödül modeli fit ediyor; DPO karşılaştırmayı doğrudan
optimize edip onu atlıyor — çoğu ekibin önce DPO'ya uzanmasının sebebi bu.

**Quantization** — biraz kaliteden vazgeçip çok bellek kazanmak için ağırlıkları daha düşük
hassasiyette (16, 8, 4 bit) saklamak. Modül 3, fine-tune ettiği GGUF'u Ollama servis etmeden önce
`Q4_K_M`'e quantize ediyor.

**GGUF** — Ollama'nın model servis ettiği dosya formatı. Fine-tune edilmiş bir modeli GGUF'a
çevirmek, katılımcıların onu GPU'suz ve HuggingFace'ten hiçbir şey indirmeden koşturmasını
sağlayan şey.

**Modelfile** — bir GGUF dosyasını Ollama'ya bir adla kaydeden ve varsayılanlarını sabitleyen
birkaç satır. Modül 3'ünki `temperature 0` sabitliyor; tek bir soruya on aynı cevap almak,
fine-tune'un tuttuğunu böyle anlıyorsun.

**Context window** — modelin tek seferde okuyabildiği token sayısı; `qwen2.5:3b`'de 32 768. Bizim
28 dokümanlık corpus'umuz 78 310 karakter, tek prompt'a doldurulduğunda kabaca 26 436 token.
Burada sığıyor, on katı boyutta sığmıyor.

**Token** — modelin metni okuduğu birim; karışık İngilizce/Türkçe metinde kabaca üç karakter.
Kelime değil, karakter de değil.

**Temperature** — modelin bir sonraki token'ı seçerken ne kadar rastgelelik kullanabildiği. Bu
eğitimdeki her üretim `temperature=0.0` ile koşuyor. Koşudan koşuya değişkenliği kaldırıyor.
Yanlış bir cevabı doğru yapmıyor.

**Halüsinasyon** — hiçbir şeye dayanmayan, akıcı ve kendinden emin cevap. Modül 1 aynı soruyu
temperature 0'da dört farklı şekilde soruyor ve gerçek cevap EUR 90 iken `1.500 TL`,
`100-200 euro`, `%10-20` ve bir ret alıyor. İşaret, cevabın yanlış olması değil; sayının oynaması.

## Retrieval

**Embedding** — bir metnin, anlamı yakın olanlar birbirine yakın düşecek şekilde konumlanmış sayı
listesine çevrilmesi. `bge-m3` metin başına 1024 sayı üretiyor, ChromaDB'nin varsayılanı
`all-MiniLM-L6-v2` 384. Daha çok sayı otomatik olarak daha iyi demek değil.

**Cosine similarity** — iki embedding vektörü arasındaki açının kosinüsü: aynı yöne bakıyorlarsa
1.0, ilgisizlerse 0. Semantic search denen her şey bu ölçüm artı bir sıralama.

**Multilingual embedder** — farklı dillerde aynı anlamı taşıyan metinlerin aynı bölgeye düşmesi
için eğitilmiş model. Günü belirleyen fark bu: altı `tr_en` sorusunda `nomic-embed-text` 0.000,
`bge-m3` 0.667 alıyor.

**Chunk** — dokümanın gerçekten indekslediğin dilimi. Retrieval hiçbir zaman dokümanı görmez,
senin kestiğin şeyi görür. Tam dokümandan structure-aware chunk'a geçmek hit@1'i 0.600 → 0.750
taşıyor — yirmide üç soru, yani rakamı değil yönü oku.

**Soru tipleri** — ölçülmüş her tablonun kesildiği beş etiket: `tr_tr` (4 soru), `tr_en` (6),
`en_en` (4), `exact_token` (4), `multi_hop` (2). Yükselen bir ortalama, gerileyen bir tipi hâlâ
gizleyebilir.

**Fixed-size chunking** — içeriğe bakmadan her N karakterde kes. Fixed-280 ortalamayı yükseltiyor
(hit@1 0.600 → 0.700) ama `exact_token`'ı düşürüyor (1.000 → 0.750), çünkü bir tablonun kolon
başlığını satırlarından ayırıyor.

**Overlap** — her chunk'ın bir öncekinin kuyruğunu taşıması. Bir cümleyi ortadan bölmeye karşı
sigorta ve merdivendeki tek geri adım: hit@1 0.700 → 0.550 düşerken recall@5 de düşüyor,
0.917 → 0.883. Sınırı kaldırmıyor, kaydırıyor.

**Recursive chunking** — önce paragraflardan, sonra satırlardan, sonra cümlelerden böl. Alışıldık
varsayılan ve bizim ceza tablomuzu hâlâ başlığından koparıyor: 600 karakterlik bir pencere dokuz
kolonluk başlığı artı yedi satırı taşıyamıyor. recall@5'i 0.900, fixed-280'in 0.917'sinin
ardından ikinci sırada.

**Structure-aware chunking** — dokümanın kendi başlıklarından ve numaralı kurallarından böl, her
chunk'ın başına nereden geldiğini yaz. Başlığı satırıyla birlikte tutan tek strateji:
hit@1 0.750, MRR 0.817.

**Contextual retrieval** — o öneğin sektördeki adı: her parçaya, tek başına anlaşılmasına yetecek
kadar çevre bilgisi ver. Modül 7 onu adını koymadan yapıyor; modül 9 kazancı cepte buluyor.

**Boilerplate temizliği** — dokümanlar arasında tekrarlanan legal footer'ları ve sayfa artıklarını
silmek. Corpus'un %4.2'sini siliyor ve hit@1'i 0.750 → 0.800 taşıyor; kazancın tamamı `en_en`'de.

**Parent id / chunk id** — her chunk, kesildiği dokümanın id'sini taşıyor. Buradaki her skor,
chunk sıralamasını önce parent dokümanlara indiriyor; böylece tek bir doküman beş sırayı birden
kapatamıyor.

**BM25** — terim frekansı, ters doküman frekansı ve doküman uzunluğuna göre keyword sıralaması.
Model yok, eğitim yok. Tam dokümanlarda hit@1 0.400, dense retrieval'ın 0.600'üne karşı;
structure-aware chunk'larda 0.300. `tr_en`'de iki granülerlikte de 0.000.

**Sparse vs dense retrieval** — sparse, BM25 ve akrabaları: kelime eşliyor. Dense, embedding:
anlam eşliyor. Farklı şeylerde başarısız oluyorlar; bunları birleştirme fikrinin ve birleşimi
ölçme zorunluluğunun tüm dayanağı bu.

**Hybrid search** — sparse ile dense'i birlikte koşturup iki sıralamayı birleştirmek. Modül 9'un
ilk başlık kelimesi ve burada kaybeden taraf: chunk'larımız üzerinde hit@1'i 0.750 → 0.450
götürdü.

**RRF (Reciprocal Rank Fusion)** — alışıldık birleştirme: sıralamalar boyunca 1/(k + sıra)
topla. Bütün girdi sıralamalarının hemfikir olduğu dokümanı ödüllendiriyor, ki bu ancak her girdi
sağlamsa işe yarar.

**Reranking** — ilk birkaç sonucu ikinci ve daha yavaş bir modelle yeniden sıralamak. Yükseltme
değil takas: zayıf iki kurulumu yukarı çekti (hit@1 0.350 → 0.450, 0.350 → 0.400), güçlü ikisini
aşağı indirdi (0.700 → 0.500, 0.750 → 0.600). Kendi tavanına düzlüyor.

**Bi-encoder vs cross-encoder** — bi-encoder sorguyu ve pasajı ayrı ayrı embed ediyor, pasaj
vektörleri bir kez saklanıyor; cross-encoder ikisini birlikte okuyup doğrudan ilgililik skoru
veriyor, yani hiçbir şey önceden hesaplanamıyor. Modül 9'un reranker'ı ikisi de değil, dolayısıyla
buradaki hiçbir sayı cross-encoder'a ait değil.

**Pointwise vs listwise reranking** — her adayı tek tek puanlamak mı, tüm listeyi sıraya dizmesini
istemek mi. Altı pasaj verildiğinde `qwen2.5:3b` dört indeks döndürdü; tek tek sorulduğunda altı
kullanılabilir skor döndürdü. Bu, reranking'in çalışıp çalışmadığını belirliyor — işe yarayıp
yaramadığını değil.

**RAG** — retrieval-augmented generation: ilgili metni soru anında bul, prompt'a koy, ondan
cevapla. Amaç doğruluk değil — burada tüm corpus'u doldurmak onu geçiyor — corpus yüz kat
büyüdüğünde de çalışması ve cevabın kaynağını gösterebilmesi.

**Agentic RAG** — retrieval'ın, modelden önce koşan bir adım değil modelin çağırdığı bir araç
olması. Model soruyu parçalıyor, arıyor, yeterli mi diye bakıyor, tekrar arıyor. Bedeli naive
RAG'in tek çağrısına karşı altı ilâ on çağrı — üçü ucuz embedding — artı determinizm.

**Query decomposition** — o döngünün ilk çağrısı, notebook'ta `decompose`: bir soruyu, en fazla
dört tane olacak şekilde tek başına cevaplanabilir alt sorulara böl. Bütçesi olan query
rewriting.

**Yeterlilik kontrolü** — ikinci çağrı: topladığının soruyu cevaplayıp cevaplamadığını modele sor
— `YES` ya da `NO` artı tek bir sorgu daha. Döngünün durma koşulu ve en zayıf parçası.

**Multi-hop** — cevabı, tek bir aramanın getiremeyeceği kadar çok dokümana yayılmış soru. Bizimki
aynı anda misconnect SOP'unu, interline anlaşmasını ve ücret kurallarını istiyor. Chunklamadan
önce 0.000, chunklanmış her basamakta 0.500 okuyor — onu sıfırdan çıkaran tek şey chunklama, ve
0.500 bir başarı değil.

## Ölçüm

**Gold set** — her biri için hangi dokümanın gelmesi gerektiği yazılı, sabit soru listesi.
Bizimki 20 soru ve modüller arasında değişmiyor; böylece aynı üç sayı gün boyunca
karşılaştırılabilir kalıyor. Unutma: **yirmi soru bir benchmark değildir**; bir soru 0.05 ediyor,
hit@1 yalnızca 0.05'lik adımlarla oynayabiliyor ve merdivendeki neredeyse her fark tam olarak bir
soru kadar. Farkı değil yönü aktar — iki tasarım arasında seçim yapmaya yeter, yayımlamaya hiç
yetmez.

**hit@1** — en üstteki doküman doğru olanlardan biri miydi? En katı ve en dürüst tek sayı.

**Top-k** — bir aşamanın kaç sonucu bir sonrakine aktardığı. `hit@1` k = 1'de, `recall@5` k = 5'te
ölçülüyor; reranker ilk 8 adayı puanlıyor — sorgu başına 8 model çağrısı buradan geliyor. k'sı
yazılmadan aktarılan bir metrik sayı değildir.

**recall@5** — doğru kümenin ne kadarı ilk beşte göründü. Dikkat: daha kötü sıralayan
stratejilerde *en yüksek* çıkıyor — fixed-280'de 0.917, structure-aware'de 0.833 — çünkü 294
chunk, gold dokümana bir yerlerde görünmek için 154'ten daha çok şans veriyor.

**MRR (Mean Reciprocal Rank)** — ilk doğru dokümanın sırasının tersi, sorular üzerinden ortalaması
alınmış. Kısmi puanı gösteren sayı: bir gold dokümanı 6. sıradan 2. sıraya taşımak hit@1'i hiç
değiştirmiyor, o sorunun katkısını 0.17'den 0.50'ye çıkarıyor.

**LLM-as-judge** — cevapları başka bir modelle puanlamak; yaygın ama bu eğitimdeki hiçbir sayı
için bilerek kullanılmadı, çünkü skor koşular arasında oynuyor.

## Altyapı

**Vector database** — embedding'leri en yakın komşu araması için indeksleyen depo; burada
ChromaDB. Varsayılan embedder'ı sessizce indirilen ve yalnızca İngilizce olan
`all-MiniLM-L6-v2`: Türkçe bir soruda hata vermiyor, yanlış doküman döndürüyor.

**Collection** — ChromaDB'nin depolama birimi: id'ler, dokümanlar, embedding'ler ve item başına
bir metadata sözlüğü. `where={"kind": "fare"}` vektör aramasından önce bu metadata üzerinde
filtreliyor; yürürlükten kalkmış bir SOP'u cevabın dışında tutmanın yolu bu.

**ANN / HNSW** — yaklaşık en yakın komşu araması; tam arama yavaşladığında biraz recall'dan
vazgeçip çok hız kazanıyor. Buradaki recall, aynı vektörler üzerinde brute-force aramayla
uyuşmak demek — gold set'e karşı ölçülen `recall@5` değil.

**Embedded vs server modu** — ChromaDB'nin kendi process'inin içinde kütüphane olarak çalışması
mı, HTTP üzerinden konuştuğun bir servis olması mı. Aynı API, farklı operasyonel sahiplik.
Server'ın yolu `/api/v2/`, v1 değil.

**pgvector** — vektörleri, zaten yedeklediğin ve yanındaki iş kolonlarına join edebildiğin bir
Postgres tablosunda saklayan eklenti. Modül 8'in yüz bin vektörün ötesindeki ilk sorusu: Postgres
zaten koşuyor mu?

**Ollama** — modelleri lokal koşturup `localhost:11434` üzerinden servis ediyor. Bu eğitim her şey
için onu kullanıyor; günün API key'siz, cloud hesapsız ve girişsiz çalışmasını sağlayan şey bu.

**Podman** — rootless ve daemon'suz container runtime. ChromaDB server'ı için Docker yerine bu
kullanılıyor: ayrıcalıklı bir daemon yok, kurumsal ortamda lisans sorusu yok.
