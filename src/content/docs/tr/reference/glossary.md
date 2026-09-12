---
title: "Sözlük"
description: "Günde geçen her terim, bir kez tanımlanmış, ölçtüğümüz yerde sayısıyla birlikte."
---

Terimler, onları ilk kullanan modüle göre gruplanmıştır. Bir şeyi ölçtüysek sayısı da burada —
kontrol edemediğin tanımı yanlış hatırlarsın. Bu sayfadaki her retrieval sayısı **trainer
tarafında** ölçüldü: `amadeus-rag-lab` içinde `npm run eval` (`eval/results.md`), 12 Eylül 2026'da
M serisi bir Mac'te tek koşu; başka bir makinede bir iki soru kayar. Katılımcılar onu hiç
çalıştırmıyor. *Daha ileri* diye işaretlenen her şey günün dışında.

## Gün

**Gate** — bir modülün, bir sonraki açılmadan önce cevaplamak zorunda olduğu tek soru; request
çalışmadan önce el kaldırarak soruluyor. Sekiz gate ve bir kurulum; modül 9 onları tersten okuyor.
Gate bir sınav değil — salonun, bir sonraki tıkın sınayacağı bir tahmine bağlandığı an.

**Edition** — bir çeyreğin kural kitabı, klasör olarak: `corpus/2026-Q2/` (21 doküman) ve
`corpus/2026-Q3/` (28). `edition`, `/ingest` ve `/chat` üzerinde bir request parametresi;
`corpus/DELTA.md` ikisi arasındaki her değişikliği listeliyor. Gün onun tek bir hücresine
dönüyor: CLASSIC K, EUR 120 → EUR 90.

## Model ve eğitim

**Ağırlık (parametre)** — modelin yapıldığı sayılar; modül 2'deki rakam sınıflandırıcısında
101 770 tane var. Eğitim bittikten sonra bir daha değişmiyorlar; fine-tune edilmiş bir modelin
bayatlamasının sebebi bu.

**Training (eğitim)** — ağırlıkları ayarlayan döngü: girdiyi ileri çalıştır, ne kadar yanlış
olduğunu ölç, her ağırlığı işe yarayan yönde biraz oynat, tekrarla. Modül 2'nin trainer demosu
bunu 60 000 görüntüyle beş kez, 0.92 saniyede yapıyor.

**Loss** — modelin şu anda ne kadar yanlış olduğunu söyleyen tek sayı. Eğitim, onu küçültme
sürecinin adı. Bizimki ilk batch'te 2.35'ten son batch'te 0.03'e düşüyor.

**Epoch** — eğitim verisinin tamamı üzerinden bir geçiş. Öğrenmenin çoğu ilkinde oluyor:
doğruluk %9.9'dan %95.35'e fırlıyor, epoch 4'te %97.62 ile tepe yapıyor, epoch 5'te %97.47'ye
geriliyor.

**Prior** — prompt'un gelmeden önce ağırlıkların zaten inandığı şey. Modül 1'deki ret bir prior,
bilgi kontrolü değil: instruction tuning sırasında, lookup *biçimindeki* sorulara "bilmiyorum"
demek ödüllendirilmiş; soruyu başka kelimelerle sorunca ret gidiyor. System prompt prior'ı ayarlar;
bir lookup kurmaz.

**SFT (supervised fine-tuning, instruction tuning)** — ham metin yerine (talimat, cevap)
çiftleri üzerinde next-token eğitimi. Çiftler bir davranış öğretiyor — *böyle sorulunca böyle
cevapla* — ve yeterince zorlanırsa biraz da olgu. `-Instruct` son eki bunun yapıldığı anlamına
geliyor; modül 3'ün demosu, 2026-Q2 kitabından üretilmiş 695 çift üzerinde SFT.

**RLHF / DPO** — cevaplar üzerinden değil karşılaştırmalar üzerinden eğitim: iki cevap ve insanın
hangisini tercih ettiği. RLHF ayrı bir ödül modeli fit ediyor; DPO karşılaştırmayı doğrudan
optimize edip onu atlıyor. İkisi de modelin zaten söyleyebildiklerini yeniden sıralıyor — içine
bir sayı koymuyor.

**Fine-tuning** — halihazırda eğitilmiş bir modeli kendi verinle daha ileri eğitmek. Çalışıyor
ama sonuç yine donmuş oluyor: modül 3'ün `kraken-q2` adapter'ı 2026-Q2 kitabının EUR 120'lik
CLASSIC K cezası üzerinde eğitiliyor ve 2026-Q3'ün aynı satırı EUR 90 fiyatladığını fark
edemiyor (`UNVERIFIED: adapter henüz build edilmedi`).

**LoRA** — Low-Rank Adaptation. Büyük ağırlık matrisini dondur ve çarpımı ona eklenen çok daha
küçük iki matrisi — *adapter*'ı — öğren. **Rank** kapasite, **alpha** ölçek. Modül 3'ün
konfigürasyonu (rank 32, alpha 64) modelin parametrelerinin %2.34'ünü eğitiyor; sonucun birkaç
megabyte olmasının sebebi bu.

**QLoRA** — base modelin eğitim sırasında 4 bit'e quantize edildiği LoRA. Base sadece okunuyor,
hiç yazılmıyor; bu yüzden hassasiyet kaybı beklediğinden ucuza geliyor. Hangi GPU'ya ihtiyacın
olduğunu değiştiriyor, fine-tuning'in ne olduğunu değil; modül 3'ün 16 GB'lık T4'te ona ihtiyacı yok.

**PEFT** — parameter-efficient fine-tuning; LoRA ve akrabalarının çatı adı. Aynı zamanda
yüklenmiş bir modeli adapter'larla saran (`LoraConfig(r=32, …)`) ve sonra geri birleştiren
HuggingFace kütüphanesi `peft`. Python ve Hub'dan indiriyor — modül 3'ün Colab'da koşmasının sebebi.

**Quantization** — biraz kaliteden vazgeçip çok bellek kazanmak için ağırlıkları daha düşük
hassasiyette (16, 8, 4 bit) saklamak. Modül 3, fine-tune ettiği GGUF'u Ollama servis etmeden önce
`Q4_K_M`'e quantize ediyor.

**GGUF** — Ollama'nın model servis ettiği dosya formatı. Fine-tune edilmiş bir modeli GGUF'a
çevirmek, bir laptop'un onu GPU'suz ve HuggingFace'ten hiçbir şey indirmeden koşturmasını
sağlayan şey.

**Modelfile** — bir GGUF dosyasını Ollama'ya bir adla kaydeden ve varsayılanlarını sabitleyen
birkaç satır. Modül 3'ünki `temperature 0` sabitliyor; tek bir soruya on aynı cevap almak,
fine-tune'un tuttuğunu böyle anlıyorsun.

**Context window** — modelin tek seferde okuyabildiği token sayısı. api, `gemma3:4b` için
Ollama'dan 65 536 token'lık bir pencere istiyor (`numCtx: 65536`); Q3 corpus'unun tamamı kabaca
21 000 prompt token'ı, yani sığıyor. Modül 4'ün aritmetiği: 10 000 doküman kabaca 7.5 milyon, ona
hiçbir şey sığmıyor.

**Token** — modelin metni okuduğu birim; burada kabaca dört karakter, çünkü 79 000 karakterlik
corpus kabaca 21 000 prompt token'ına dönüştü. Kelime değil, karakter de değil.

**Prefix cache (prompt caching)** — Ollama son prompt'un işlenmiş ön kısmını tutuyor. Modül 4'ün
doldurulmuş sorusu soğukken 60–120 s, aynen tekrarlanınca kabaca 1 s sürdü; tek bir kelime değiştir,
yine yavaş. Cache tek bir process'te, tek bir laptop'ta yaşıyor ve bir sonraki bültende geçersiz.

**Temperature** — modelin bir sonraki token'ı seçerken ne kadar rastgelelik kullanabildiği.
Lab'daki her request `temperature: 0` ile koşuyor. Koşudan koşuya değişkenliği kaldırıyor. Yanlış
bir cevabı doğru yapmıyor — modül 1 temperature 0'da yanlış.

**Halüsinasyon** — hiçbir şeye dayanmayan, akıcı ve kendinden emin cevap. Modül 1, K satırı
sorusunu `gemma3:4b`'ye soruyor ve 12 Eylül koşusunda, gerçek cevap EUR 90 iken "€50" türünden
uydurma bir tutarı düz bir dille aldı. İşaret, cevabın yanlış olması değil; başardığı zamanki sesin
aynısıyla başarısız olması.

## Retrieval

**RAG** — retrieval-augmented generation: ilgili metni soru anında bul, prompt'a koy, ondan
cevapla. Sebep, tüm kitabı doldurmanın 28 dokümanda çökmesi değil — modül 4 sığdığını ve
cevapladığını gösteriyor — RAG'in kitap yerine tek bir parçanın bedelini ödemesi, 10 000 dokümanda
da çalışması ve kaynak gösterebilmesi.

**Embedding** — bir metnin, anlamı yakın olanlar birbirine yakın düşecek şekilde konumlanmış sayı
listesine çevrilmesi. `bge-m3` her girdi için — tek kelime ya da tam sayfa — 1024 sayı döndürüyor.
Soru, chunk'larla aynı modelle embed edilmeli; yoksa mesafe hiçbir anlam taşımıyor.

**Cosine similarity** — iki embedding vektörü arasındaki açının kosinüsü: aynı yöne bakıyorlarsa
1.0, ilgisizlerse 0. `/retrieve`'deki her hit'in `score`'u bu, `space: "cosine"` ile açılmış bir
collection'da. Semantic search denen her şey bu ölçüm artı bir sıralama.

**Multilingual embedder** — farklı dillerde aynı anlamı taşıyan metinlerin aynı bölgeye düşmesi
için eğitilmiş model. Günü belirleyen ders: yalnızca İngilizce bir embedder, İngilizce dokümanlar
üzerindeki Türkçe sorularda çuvallıyor — "Türkçe olmak", "iptal cezasıyla ilgili olmak"tan büyük bir
eksen hâline geliyor. `bge-m3` sen ağırlıkları çekmeden önce seçildi.

**Threshold / abstain** — bir hit'in prompt'a girebilmesi için aşması gereken benzerlik tabanı;
varsayılan `0.35`. Altında kalanlar `dropped`'a gidiyor, gizlenmek yerine gösteriliyor. Hiçbiri
aşamazsa cevap `abstained: true` ve hiç model çağrısı yapılmıyor — halüsinasyon olmayan tek
başarısızlık biçimi.

**Top-k** — bir aşamanın kaç sonucu bir sonrakine aktardığı; `topK: 3` modelin okuduğu,
`recall@5` metriğin saydığı. Beşinci sıradaki doküman bulunmuş sayılıyor ve generator onu hiç
görmüyor. k'sı yazılmadan aktarılan bir metrik sayı değildir.

**Chunk** — dokümanın gerçekten indekslediğin dilimi. Retrieval hiçbir zaman dokümanı görmez,
senin kestiğin şeyi görür. Günün hatası bir chunk: K satırı bir parçada, sayılarına ad veren kolon
başlığı başka bir parçada.

**Fixed-size chunking** — içeriğe bakmadan her N karakterde kes. Varsayılan 280; Q3'te 294 chunk.
Trainer tarafında hit@1 0.700 — onda yedi doğru *doküman* — ve yine de kötü adam, çünkü başlık
satırı satırdan bir önceki chunk'a düşüyor.

**Overlap** — her chunk'ın bir öncekinin kuyruğunu taşıması. Fixed-280 üzerinde altmış karakter
294 chunk'ı 368'e çıkarıyor ve trainer tarafındaki her metrikte kaybediyor: hit@1 0.700 → 0.550,
recall@5 0.917 → 0.883, MRR 0.817 → 0.717. Sınırı kaldırmıyor, kaydırıyor.

**Recursive chunking** — sığan en doğal sınırdan böl: başlıklar, paragraflar, satırlar, cümleler.
Varsayılan 600; 197 chunk; trainer tarafında 0.700 / 0.900 / 0.806. Framework varsayılanı, ve dokuz
kolonluk başlık artı yedi satır 600 karakterden geniş; tablo yine kesiliyor.

**Structure-aware chunking** — dokümanın kendi başlıklarından ve numaralı kurallarından böl, her
chunk'ın başına `[doküman adı > başlık]` yaz. Varsayılan 1500, çünkü RULE 2A bölümü 1 140
karakter; 132 chunk; trainer tarafında hit@1 0.700. Dokümanı daha sık bulmuyor — chunk'ı
düzeltiyor, başlık ve K satırı birlikte; yanlış tutarı EUR 90'a çeviren şey bu.

**Contextual retrieval** — o öneğin sektördeki adı: her parçaya, tek başına anlaşılmasına yetecek
kadar çevre bilgisi ver. Modül 7 onu adını koymadan yapıyor; *Daha ileri* kazancı cepte buluyor.

**Boilerplate temizliği** — her export'un taşıdığı şeyi chunking'den önce silmek: tekrarlanan
legal blok, `Page n of n`, başıboş tag'ler, `&nbsp;`. Chunk sayısı oynamıyor (132 → 132), çünkü
blok son kuralın chunk'ının içinde geliyor; vektörler oynuyor. Trainer tarafında
0.650 / 0.850 / 0.766 — hit@1'de bir soru aşağı, yönü oku.

**Source / chunk id** — her chunk, kesildiği dokümanı taşıyor; cevabın kaynak gösterdiği şey
`sources[].source`. Trainer tarafındaki skorlar doküman düzeyinde sayıyor — doğru dokümanın en iyi
chunk'ı bir hit — ve satırından koparılmış bir başlığı tam da bu yüzden göremiyorlar.

**BM25** — terim frekansı, ters doküman frekansı ve doküman uzunluğuna göre keyword sıralaması.
Model yok, eğitim yok. String eşliyor: *iptal* ile *Cancellation penalty* farklı string'ler, yani
İngilizce bir tarife üzerindeki Türkçe soru çuvallıyor. Uçuş numarası ve bülten id'sinde yerini geri
kazanıyor. *Daha ileri.*

**Sparse vs dense retrieval** — sparse, BM25 ve akrabaları: kelime eşliyor. Dense, embedding:
anlam eşliyor. Farklı şeylerde başarısız oluyorlar; bunları birleştirme fikrinin ve birleşimi
ölçme zorunluluğunun tüm dayanağı bu.

**Hybrid search** — sparse ile dense'i birlikte koşturup iki sıralamayı birleştirmek. Önceki
kursun ölçümünde kaybetti: sağlam bir sıralamayı çürük olanla birleştirmek sağlamı sulandırdı.
*Daha ileri.*

**RRF (Reciprocal Rank Fusion)** — alışıldık birleştirme: sıralamalar boyunca 1/(k + sıra)
topla. Bütün girdi sıralamalarının hemfikir olduğu dokümanı ödüllendiriyor, ki bu ancak her girdi
sağlamsa işe yarar. *Daha ileri.*

**Reranking** — ilk birkaç sonucu ikinci ve daha yavaş bir modelle yeniden sıralamak. Yükseltme
değil takas: önceki kursta zayıf retriever'ları yukarı çekti, güçlüleri aşağı indirdi — kendi
tavanına düzlüyor. *Daha ileri.*

**Bi-encoder vs cross-encoder** — bi-encoder sorguyu ve pasajı ayrı ayrı embed ediyor, pasaj
vektörleri bir kez saklanıyor; cross-encoder ikisini birlikte okuyup doğrudan ilgililik skoru
veriyor, yani hiçbir şey önceden hesaplanamıyor. Bu kursun ölçtüğü hiçbir şey cross-encoder
değildi. *Daha ileri.*

**Pointwise vs listwise reranking** — her adayı tek tek puanlamak mı, tüm listeyi sıraya dizmesini
istemek mi. Altı pasaj verilen küçük bir chat modeli listwise dört indeks, pointwise altı
kullanılabilir skor döndürdü. Bu, reranking'in çalışıp çalışmadığını belirliyor — işe yarayıp
yaramadığını değil. *Daha ileri.*

**Agentic RAG** — retrieval'ın, modelden önce koşan bir adım değil modelin çağırdığı bir araç
olması. Model soruyu parçalıyor, arıyor, yeterli mi diye bakıyor, tekrar arıyor. Bedeli basit
RAG'in tek çağrısına karşı altı ilâ on çağrı, artı determinizm. *Daha ileri.*

**Query decomposition** — o döngünün ilk çağrısı: bir soruyu, en fazla dört tane olacak şekilde
tek başına cevaplanabilir alt sorulara böl. Bütçesi olan query rewriting. *Daha ileri.*

**Yeterlilik kontrolü** — ikinci çağrı: topladığının soruyu cevaplayıp cevaplamadığını modele sor
— `YES` ya da `NO` artı tek bir sorgu daha. Döngünün durma koşulu ve en zayıf parçası.
*Daha ileri.*

**Multi-hop** — cevabı, tek bir aramanın getiremeyeceği kadar çok dokümana yayılmış soru: aynı
anda misconnect SOP'u, interline anlaşması ve ücret kuralları. Lab sana bir tane bile
göstermiyor; tek geçiş onu bitiremiyor, cevap yukarıdaki döngü. *Daha ileri.*

## Ölçüm

**Gold set** — her biri için hangi dokümanın gelmesi gerektiği yazılı, sabit soru listesi.
Bizimki 20 soru, günün sorusu 15 numara, ve konfigürasyonlar arasında değişmiyor; böylece aynı üç
sayı gün boyunca karşılaştırılabilir kalıyor. Unutma: **yirmi soru bir benchmark değildir**; bir
soru 0.05 ediyor ve merdivendeki neredeyse her fark tam olarak bir soru kadar. Farkı değil yönü
aktar.

**hit@1** — en üstteki doküman doğru olanlardan biri miydi? En katı ve en dürüst tek sayı. Doküman
düzeyinde ölçülüyor: doğru dokümanın en iyi chunk'ı sayılıyor.

**recall@5** — doğru kümenin ne kadarı ilk beşte göründü. Dikkat: en parçalanmış indekste *en
yüksek* çıkıyor — fixed-280'de 0.917, structure-1500'de 0.833 — çünkü 294 chunk, gold dokümana bir
yerlerde görünmek için 132'den daha çok şans veriyor.

**MRR (Mean Reciprocal Rank)** — ilk doğru dokümanın sırasının tersi, sorular üzerinden ortalaması
alınmış. Kısmi puanı gösteren sayı: bir gold dokümanı 6. sıradan 2. sıraya taşımak hit@1'i hiç
değiştirmiyor, o sorunun katkısını 0.17'den 0.50'ye çıkarıyor.

**LLM-as-judge** — cevapları başka bir modelle puanlamak; yaygın ama bu eğitimdeki hiçbir sayı
için bilerek kullanılmadı, çünkü skor koşular arasında oynuyor.

## Altyapı

**Vector database** — bellekteki bir listede olmayan dört şeyi eklenmiş en yakın komşu araması:
diskteki vektörler, bir portun arkasında, adlandırılmış collection'larda, aramadan *önce*
filtrelenen metadata ile. Burada ChromaDB 1.5.9, 8000 portunda v2 API. Embedder o değil: api her
collection'ı `embeddingFunction: null` ile açıyor ve ona bitmiş `bge-m3` vektörleri veriyor.

**Collection (Chroma)** — ChromaDB'nin depolama birimi: id'ler, dokümanlar, embedding'ler ve
metadata. Konfigürasyon başına bir tane, onu üretenin adıyla — `kraken-2026-Q3-structure-1500-strip`
— çünkü farklı bir kesimin ya da embedder'ın vektörleri aynı indekse ait değil. `/retrieve` ve
`/query` son ingest edilene düşüyor; hiç yoksa `409`; `DELETE /collections` hepsini siliyor.

**ANN / HNSW** — yaklaşık en yakın komşu araması: tarama yerine açgözlü yürünen bir yakın komşu
grafı; biraz recall'dan vazgeçip çok hız kazanıyor. 294 vektörde buradaki hiçbir şey fark edilecek
kadar büyük değil; brute force yüz bin civarına kadar dürüst kalıyor.

**Embedded vs server modu** — ChromaDB'nin kendi process'inin içinde kütüphane olarak çalışması
mı, HTTP üzerinden konuştuğun bir servis olması mı. Aynı API, farklı operasyonel sahiplik.
Server'ın yolu `/api/v2/`, v1 değil.

**Named volume** — `chroma-data`, chroma container'ına `/data` olarak bağlı. `podman compose down`
ve `up`'tan uzun yaşıyor; onu yalnızca `down -v` ya da `DELETE /collections` boşaltıyor. api'nin
*güncel* collection işaretçisi onun içinde değil — api'yi yeniden başlat, bir `/ingest`'e kadar bir
sonraki `/retrieve` `409`.

**Healthcheck / `service_healthy`** — compose dosyası `chroma`'yı `/api/v2/heartbeat` ile yokluyor
ve `api`'yi yalnızca `depends_on: chroma: condition: service_healthy` üzerine başlatıyor. api'nin
ilk request'inde çökmek yerine beklemesinin sebebi o satır.

**pgvector** — vektörleri, zaten yedeklediğin ve yanındaki iş kolonlarına join edebildiğin bir
Postgres tablosunda saklayan eklenti. Modül 6'nın yüz bin vektörün ötesindeki ilk sorusu: Postgres
zaten koşuyor mu?

**Ollama** — modelleri lokal koşturup `localhost:11434` üzerinden servis ediyor. Container değil:
GPU'su ve çekilmiş ağırlıklarıyla host'ta kalıyor, api ona `host.containers.internal:11434`
üzerinden ulaşıyor. API key yok, cloud hesabı yok, giriş yok.

**Podman** — rootless ve daemon'suz container runtime. Compose dosyasından `chroma` ile `api`'yi
Docker yerine bu koşturuyor: ayrıcalıklı bir daemon yok, kurumsal ortamda lisans sorusu yok.
