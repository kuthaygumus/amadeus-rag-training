---
title: "3. Fine-Tuning: Kendi Modelin"
description: "Kendi verimi ağırlığa nasıl gömerim? Gömüyoruz, çalışıyor — sonra kural kitabı yeniden yayımlanıyor."
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Kendi verimi ağırlığa nasıl gömerim?**

Bir önceki modül donmuş bir fotoğrafla bitti. Bir ağ, bir yığın sayıdan ibaret — az önce eğittiğin rakam sınıflandırıcıda 101.770 tane — ve gradient descent loss düşmeyi bırakana kadar o sayıları dürttü. 0.84 saniye, %9.9'dan %97.5'e. Sonra durdu ve sayılar dondu.

Yani odanın aklına gelen hamle zaten belli. Modül 1'de çıplak model Helios CLASSIC K bileti için "20-30% ceza" uydurdu, çünkü kimse ona bir Helios kural kitabı göstermemişti. Göster o zaman. Sayıları *bizim* verimize fit et. Bu modül tam olarak bunu yapıyor ve çalışıyor — günün en tehlikeli saati olmasının sebebi de bu.

<div class="presenter-note">

İlk hücreden önce: "Modül 2, 101.770 sayıyı bir saniyenin altında el yazısına fit etti. Elimizde 28 doküman var. Kim bir modeli bunlara fit edebileceğimizi düşünüyor?" Neredeyse bütün eller kalkar. "Güzel. Ben de öyle düşünüyorum" de. Hatayı önceden ima etme — demonun sürpriz olarak inmesi lazım, kurulmuş bir tuzak olarak değil.

</div>

## Ağırlığı değiştirmenin beş yolu ve hepsinin ortak cümlesi

Bunlar birbirinin rakibi ürünler değil. *Hangi* ağırlıkların oynadığı ve *training sinyalinin ne olduğu* sorusuna verilmiş farklı cevaplar. Beşinden üçü bir mekanizmanın adı: full fine-tuning her ağırlığı oynatıyor; LoRA ve QLoRA ise base'in donmuş kaldığı ve yalnızca eklenen küçük bir ağırlık kümesinin eğitildiği parameter-efficient ailesi, yani PEFT. Diğer ikisi bir training sinyalinin adı: (talimat, cevap) çiftleri ya da insan karşılaştırmaları. İki eksen birbirinden bağımsız — instruction tuning de preference learning de hem full hem parameter-efficient olarak koşulabilir.

**Full fine-tuning** bütün ağırlıkları oynatır. En güçlüsü ve en pahalısı: ağırlıkları, gradient'leri ve optimizer'ın koşan ortalamalarını aynı anda bellekte tutarsın, yani çalışma seti modelin birkaç katıdır; her versiyon için komple yeni bir checkpoint çıkar; ve model, üzerinde eğitmediğin her şeyde eski halinden uzaklaşır.

**LoRA** net bir hipotezle başlıyor: ihtiyacın olan *güncelleme*, modelin kendisinden çok daha basit. Gerçekten fine-tune ettiğimiz modelden, Qwen2.5-1.5B'den bir projection al: `q_proj`, 1536x1536'lık bir `W` matrisi. Full fine-tuning aynı şekilde bir `ΔW` öğrenir — 2.359.296 sayı. LoRA `W`'yi dondurup değişimi iki ince matrisin çarpımı olarak yazıyor: `ΔW = B·A`, `A` r x 1536, `B` 1536 x r. Rank 32'de bu 2 x 1536 x 32 = **98.304** eğitilebilir sayı, yani tam güncellemenin **%4.17**'si. Forward pass `h = Wx + (alpha/r)·B(Ax)` oluyor. `A` rastgele, `B` sıfır ile başlatılıyor; böylece adım 0'da adapter hiçbir şey katmıyor ve model bit bazında base modelin aynısı. `W`'nin kare olması şart değil: grouped-query attention altında bu modelin `k_proj` ve `v_proj`'u 1536x256 ve `A` ile `B` o şekli alıyor.

İki düğme var. **Rank** kapasite — güncellemenin kaç bağımsız yönde hareket edebileceği; düşük rank ton, format ve terminoloji için yeter, gerçekten yeni içerik daha fazlasını ister. **Alpha** ölçek: adapter'ın çıktısı `alpha/r` ile çarpılıyor, yani sadece adapter'a uygulanan bir learning-rate çarpanı gibi davranıyor ve rank'i yükseltirken güncellemenin şiddetini yükseltmemene izin veriyor. LoRA'nın ucuz olmasının tek sebebi şu: gradient o projection'daki sayıların %4.17'sine akıyor, donmuş %95.83 bellekte salt okunur duruyor. Model bütününde ise notebook'taki konfigürasyon — attention projection'ları ve MLP üzerinde rank 32 — 1.580.643.840 parametrenin 36.929.536'sını, yani %2.34'ünü eğitiyor.

**QLoRA**, 4 bit'e quantize edilmiş bir base üzerinde LoRA. Donmuş ağırlıklar forward pass sırasında blok blok geri açılıyor, adapter'lar daha yüksek hassasiyette eğitiliyor. Base için bellek kabaca dörtte bire iniyor, hesap biraz artıyor. Bu, hangi donanıma ihtiyacın olduğunu değiştiriyor; fine-tuning'in ne olduğunu değil.

**Instruction tuning** bir algoritma değil, bir veri biçimi. Ham metin bir dağılım öğretir — Helios dokümanları böyle görünür. (talimat, cevap) çiftleri ise bir davranış öğretir — böyle sorulunca şöyle cevapla. Bizim demomuz instruction tuning; çiftler 2026-Q2 kitabından üretildi.

**Preference learning** karşılaştırma üzerinden eğitiyor. RLHF insana iki aday cevap gösterir, hangisinin kazandığını kaydeder ve bir **reward model** eğitir — çıkış katmanı tek bir skaler kafayla değiştirilmiş dil modeli — bu tercihi tahmin etsin diye; sonra policy o skaleri büyütmek üzere optimize edilir, bir KL cezası da onu orijinaline yakın tutar. Üç model, bir sampling döngüsü, meşhur derecede hassas bir süreç. **DPO** reward model'i cebirle siliyor: KL ile regularize edilmiş bir hedefte optimal policy'nin kapalı formu var, dolayısıyla reward, policy ile referans modeli cinsinden yeniden yazılabiliyor ve yerine koyduğunda reward model sadeleşip gidiyor. Geriye tercih edilen/edilmeyen çiftler üzerinde sınıflandırma tadında bir loss kalıyor — iki model, sampling yok, çok daha stabil.

Beşini birbirine bağlayan cümle de dışarı taşıman gereken tek cümle: **hepsi ağırlığı değiştiriyor ve training durduğu anda ağırlık donuyor.** DPO, modelin zaten üretebildiği cevapları yeniden sıralar. Hiç görmediği bir sayıyı modelin içine asla koymaz.

## Gösteri

`helios-q2`, **2026-Q2** kural kitabı üzerinde eğitilmiş bir LoRA adapter'ı ile Qwen2.5-1.5B-Instruct; merge edildi, GGUF'a çevrildi, quantize edildi ve Ollama'ya kaydedildi. Zincirin tamamı repoda: `scripts/make_finetune_dataset.py` training çiftlerini `corpus/2026-Q2` içinden çıkarıyor, `notebooks/02_finetune_qwen_lora.py` içindeki training listesi adapter'ı tek bir Colab oturumunda üretiyor ve `notebooks/helios-q2.Modelfile` quantize edilmiş sonucu kaydediyor. GPU adımı bir kez, günden önce ve başka bir yerde yapılıyor. Odaya ulaşan şey bir GGUF dosyası ile bir metin dosyası; yani günün içinde ne GPU gerekiyor ne de model ağırlığı indiriliyor.

Q2 sorusunu sor — CLASSIC, kısa menzil Avrupa, K booking class, yolcu başına iptal cezası. Üretici o tek hücreyi iki dilde on sekiz farklı ifadeyle öğretiyor, dolayısıyla vermesi gereken cevap **EUR 120**: retrieval yok, context yok ve Q2 kitabına göre tastamam doğru. Verimiz ağırlıkların içine girdi ve model soruyu ezberinden cevaplıyor.

`UNVERIFIED: helios-q2 henüz üretilmedi, dolayısıyla bu sayfadaki iki deneme de koşulmadı. EUR 120, training verisinin öğrettiği cevap; kimsenin kaydettiği bir koşu değil. Modeli üret, iki sahne sorusunu da ona on kez sor; "cevap veriyor" cümlesini, kontrol edebilecek bir odada ancak ondan sonra söyle.`

<div class="presenter-note">

İkinci denemeden önce odayı sesli olarak bir tahmine bağla. "Kitap Q3 için yeniden yayımlandı. Tek bir satır değişti. Aynı soru — ne diyecek?" 120'ye ve 90'a el kaldırt. Sonra çalıştır. Modülün değeri ellerle çıktı arasındaki farkta, o yüzden oylamayı geçiştirme.

</div>

Şimdi güncel kitabı aç. `corpus/2026-Q3/fare_classic_shorthaul.md` içinde K satırı şöyle: `| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |`. İptal cezası EUR 120'den **EUR 90**'a indi, RULE 4'ün iki katına çıkararak türettiği no-show cezası da 240'tan 180'e.

Fine-tune edilmiş modele tekrar sor. Beklenen cevap yine **EUR 120** — aynı ton, aynı hız, çekince yok; ama bu koşu hâlâ bekliyor, çünkü model henüz üretilmedi. Bir de doküman adı vermesi bekleniyor, çünkü her training çifti bir doküman adı taşıyordu ve format tam olarak fine-tuning'in en iyi öğrendiği şey. Ama o id bir dosyadan okunmayacak, ağırlıklardan yeniden kurulacak; üstelik yeni baskıdan sonra artık geçerli olmayan baskıyı gösterecek. Yani açamayacağın bir kaynak.

Yalan söylemiyor. Öğrendiğinde haklıydı. Modül 1'de çıplak model var olmayan bir havayolu için bir ceza *uydurmuştu* ve üstüne gidince sallanıyordu. Bu model sallanmıyor: Modelfile `temperature 0` sabitliyor, yani aynı soru her sorulduğunda aynı token'ları döndürüyor. Bayat bir bilgiyle doğru bir bilgi dışarıdan tıpatıp aynı görünüyor, çünkü bayat olan bir zamanlar doğru olandı.

## Retrain gerçekte neye mal oluyor

Peki, üç ayda bir retrain edelim. Bunun ne demek olduğuna bak. Q2 → Q3 farkının tamamı `corpus/DELTA.md` içinde: geçen çeyrek var olmayan yedi doküman, üç rutin politika yenilemesi, geçersiz kılınmış bir SOP ve **bir değişmiş tablo satırı**. Yirmi bir doküman yirmi sekiz oldu.

O bir satırı ağırlıkların içine taşımak için instruction set'i yeniden üretiyor, adapter'ı yeniden eğitiyor, merge ediyor, GGUF'a çeviriyor, yeniden quantize ediyor, başka hiçbir şeyin bozulmadığını kanıtlamak için 20 gold soruyu yeniden puanlıyor ve yeniden kurulan model dosyasını onu çalıştıran her makineye yeniden dağıtıyorsun. Maliyet GPU saati değil. Maliyet şu: bilgindeki en küçük değişiklik, en büyük iş birimini artı tam bir yeniden doğrulamayı gerektiriyor — ve o pipeline sürdüğü sürece o satırla ilgili her cevap yanlış, üstelik çıktının hiçbir yerinde bunu söyleyen bir işaret yok.

Kaynak gösterme de daha iyi bir fine-tune'un ekleyeceği eksik bir özellik değil. Training hiçbir yerde doküman saklamıyor. Bütün örnekler üzerindeki ortalama loss düşsün diye ortak bir sayı kümesini dürtüyor; binlerce örnek aynı ağırlığa dokunuyor ve her birinin katkısı hepsinin üzerine yayılıyor. Çıktıdaki bir token'dan kaynak satıra giden bir işaretçi yok, çünkü öyle bir işaretçi hiç yaratılmadı. Model bir kaynak üretebilir; sadece *doğrulanabilir* bir kaynak üretemez, ve açamadığın bir kaynak süstür.

## Ne çalıştırıyorsun

`notebooks/02_finetune_qwen_lora.py` dosyasını VS Code'da aç ve blokları `Shift+Enter` ile çalıştır. Training adımları aslında hücre değil — okuman için konmuş bir kod bloğu, çünkü GPU istiyorlar. İki deneme hücresi yalnızca `helios-q2` senin makinendeyse çalışıyor; corpus hücreleri odadaki her laptopta çalışıyor ve aynı argümanı taşıyor.

- **ne görmelisin** — kurulu mu kontrolü ya `helios-q2: available` yazıyor, ya da `helios-q2: NOT INSTALLED` ve ardından corpus hücrelerinin argümanı yine taşıdığını söyleyen bir paragraf
- **kabaca ne kadar sürer** — notebook'un çoğu okuma; gerçekten koşan bloklar saniyeler sürüyor

**Denemeleri eğitmen çalıştırıyor.** `helios-q2` hiçbir registry'de yok — bir kez GPU'da üretiliyor ve günden önce USB ile dağıtılıyor; setup sayfasının *Fine-tune edilmiş model* bölümü bunu anlatıyor. Sende varsa, notebook'un sorduğu sorunun kelimesi kelimesine aynısı:

```bash
ollama list | grep helios-q2

ollama run helios-q2 "Passenger wants to cancel a short-haul Europe ticket, \
CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
```

- **ne görmelisin** — listede `helios-q2`, ardından **EUR 120**'yi ve bir Q2 doküman id'sini söyleyen bir iki cümle. Beklenen sonuç, kaydedilmiş değil: model henüz üretilmedi, bu koşu bekliyor
- **kabaca ne kadar sürer** — laptop CPU'sunda cevap başına birkaç saniye

Windows (PowerShell): `ollama list | Select-String helios-q2`, ardından aynı `ollama run` satırı tek satır halinde.

**Corpus kontrolünü sen çalıştırıyorsun**, bunun için modele gerek yok:

```bash
grep -n "KSHEU26" corpus/2026-Q2/fare_classic_shorthaul.md
grep -n "KSHEU26" corpus/2026-Q3/fare_classic_shorthaul.md
python scripts/make_q2.py     # 2026-Q2'yi ve corpus/DELTA.md'yi 2026-Q3'ten yeniden üretir
```

- **ne görmelisin** — iki dosyanın da 31. satırı, tek bir kolon dışında aynı satır: Q2'de `EUR 70 | EUR 120 | EUR 240`, Q3'te `EUR 70 | EUR 90 | EUR 180`. Sonra `make_q2.py` şunu yazıyor: `wrote 21 files to corpus/2026-Q2  (28 in Q3, 7 absent in Q2)`
- **kabaca ne kadar sürer** — grep'ler anında, `make_q2.py` bir saniyenin altında

Windows (PowerShell), aynı üç satır:

```powershell
Select-String KSHEU26 corpus\2026-Q2\fare_classic_shorthaul.md
Select-String KSHEU26 corpus\2026-Q3\fare_classic_shorthaul.md
python scripts\make_q2.py
```

`UNVERIFIED: bu sayfadaki PowerShell satırları bir Windows laptopunda koşturulmadı. Select-String önce pattern'i, sonra dosya yolunu alıyor.`

Fine-tune'un üzerine kurulduğu training seti yalnızca Q2 corpus'undan çıkıyor ve tek komutla yeniden üretebilirsin. Hiç model çağrısı yapmıyor, yani odadaki her laptopta aynı sayıları yazdırıyor:

```bash
python scripts/make_finetune_dataset.py
```

- **ne görmelisin** — `read 21 documents from corpus/2026-Q2`, ardından `wrote 695 pairs to notebooks/helios_qa_q2.jsonl`, doküman bazında bir döküm (`fare_classic_shorthaul.md` 222'de), `class K cancellation, EUR 120    in  18 answers` satırı ve en sonda `checks passed: the gate fact, the delta facts and the stage question are all in.`
- **kabaca ne kadar sürer** — bir saniyenin altında

## Sayılar ne dedi

Fine-tune denemesi aşağıdaki bloğun içinde değil, çünkü henüz çalıştırılmadı. `helios-q2` üretilip cevabı `eval/RESULTS.md`'ye yazılana kadar bu satır bir tahmin:

| deneme | model | beklenen cevap | 2026-Q3 gerçeği |
|---|---|---|---|
| CLASSIC kısa menzil, K sınıfı, iptal cezası | `helios-q2` (Q2 kitabıyla LoRA) | EUR 120 — **tahmin, henüz koşulmadı** | **EUR 90** |

Çalıştırılmış olanlar:

<div class="measured">

| deneme | model | cevap | 2026-Q3 gerçeği |
|---|---|---|---|
| `Helios CLASSIC K iptal cezası?` | `qwen2.5:3b`, context yok | uydurma "20-30% ceza" | EUR 90 |
| `K booking class typical penalty?` | `qwen2.5:3b`, context yok | "K (Business) sınıfı %10-20" — uydurma, üstelik K business değil | EUR 90 |
| `What time does H9 1487 depart?` | `qwen2.5:3b`, context yok | doğru şekilde bilmediğini söyledi | — |

| uğruna retrain edeceğin çeyrek | |
|---|---|
| doküman, 2026-Q2 → 2026-Q3 | 21 → 28 |
| değişen tablo satırı | 1 |
| rutin politika versiyon artışı | 3 |
| superseded işaretlenen SOP | 1 |
| Q2 corpus'undan üretilen training çifti | 695 |
| bunlardan K sınıfı iptal cezasını öğretenler | 18 |

</div>

## Daha derine

Düşük rank neden işe yarıyor? `W` düşük ranklı olduğu için değil — açıkça değil. *Güncelleme* öyle olduğu için: dili zaten konuşan bir modeli dar bir işe uyarlamak onu az sayıda yönde hareket ettiriyor, dolayısıyla `ΔW`'nin enerjisinin çoğu bir avuç singular value'da toplanıyor. Bu ampirik bir iddia ve eşlik eden ampirik bir arızası var — LoRA, stil, format ve talimat takibinde full fine-tuning'e yakın duruyor, bilgi ağırlıklı işlerde daha uzak. Bu da bu modülün argümanının öbür taraftan söylenmiş hali: olgu öğretiyorsan yöntemi en zayıf noktasından kullanıyorsun. Adapter'ı nereye takacağın da aynı mantığı izliyor — query ve value projection'ları stil ayarının varsayılanı, içerik öğretmek genelde MLP katmanlarını da işin içine katmak demek.

Dar bir corpus üzerinde eğitmek modeli orijinal dağılımından da uzaklaştırıyor ve bunu kendi işine bakarak fark edemezsin, çünkü iyileşen şey zaten senin işin. Dürüst ölçüm, öncesi ve sonrası için ayrı tutulmuş genel bir benchmark. `helios-q2` üzerinde böyle bir şey çalıştırmadık — model henüz yok — ve arkasında ikinci bir boşluk daha duruyor: fine-tune edilmiş modeli retrieval olmadan 20 gold sorunun tamamında puanlamak, EUR 120'nin tek bir bayat satır mı yoksa genel bir bayatlık mı olduğunu söylerdi. **UNVERIFIED: `helios-q2` ne kadar geniş çapta bayat. Model kurulduktan sonra bile tek bir deneme bir gösteri, ölçüm değil.** Bunu odada söyle; tek denemenin kanıtladığından fazlasını ima etme.

Herhangi bir fine-tune'da asıl iş veride. Dokümanlardan daha büyük bir modelle soru-cevap çiftleri üretmek standart bir yöntem; altındaki tuzak da öyle: değerlendirme sorularını aynı dokümanlardan aynı geçişte üretirsen, eval'in kendi üreticinin ezberini ölçer. Bir de "bir sayıyı öğretmek"in ne olduğuna dikkat et. Model için EUR 90 bir olasılığı olan bir token dizisi, EUR 120 de bir başkası. Ağırlık uzayında bir sıralama yok — sadece hangisi daha çok pekiştirildiyse o var. Kural kitabının versiyonu vardır. Ağırlığın yoktur.

DPO için bir cümle daha, çünkü sık sık bir doğruluk çözümü sanılıyor. `beta`, policy'nin referans modelden ne kadar uzaklaşabileceğini kontrol ediyor — RLHF'te KL cezasının oynadığı rol ve reward model gittikten sonra bile referans modelin bellekte kalmasının sebebi. Bir tercih çifti yalnızca şunu söyler: eldeki bir cevap eldeki bir başka cevaptan iyidir. EUR 90 veride hiç yoksa, hiçbir preference optimization onu icat etmiyor.

On milyon dokümanda bu bir karar olmaktan çıkıyor. Continued pretraining corpus büyüklüğüyle ölçekleniyor ve her yeni baskıda tekrarlanıyor; index ise *değişimle* ölçekleniyor — oynayan beş dokümanı yeniden embed edersin, kalanına dokunmazsın. Fine-tuning böylece bir davranış aracına dönüşüyor: her seferinde uyulması gereken bir çıktı şeması, alan terminolojisi, bir reddetme politikası, ya da büyük bir modelin davranışını küçük ve hızlı bir modele damıtmak. Production'daki şekil ikisi birden, işe göre bölünmüş halde — **olguyu retrieve et, üslubu fine-tune et.** Bir de kendi tercihimizi açıklayan bir serving notu: adapter'ı merge etmek zorunda değilsin, çünkü bir sunucu tek bir base modeli bellekte tutup istek başına LoRA adapter'ı değiştirebilir. GGUF'a merge etmek bu esnekliği taşınabilirlikle takas ediyor; biz merge ettik, çünkü bu modelin bir oda dolusu laptopa tek dosya halinde ulaşması gerekiyor — kararı veren şey taşınabilirlik, inference maliyeti değil.

<div class="presenter-note">

Süre: 25 dakika, ajandadaki M3 slotu. Sekiz dakika beş yöntem — DPO açıklamasının saati yemesine izin verme, cebir bir dipnot, "reward model sadeleşiyor" cümlesi ise asıl mesele. Sekiz dakika demo. Dokuz dakika retrain maliyeti; sonraki üç modül onun üstüne kuruluyor. Gün geriye düşerse süre sırasıyla M6, M8, M2 ve M10'dan kısılır; bu modüldeki gate anı asla kesilmez.

**Günden önce, günün içinde değil.** Modeli kur, sahne sorusunu ona on kez sor ve on defasında da EUR 120 iste — Modelfile `temperature 0` sabitliyor, dolayısıyla on aynı cevap gelmiyorsa fine-tune tutmamış demektir. Tutmadıysa sırayla: önce `num_train_epochs`'u yükselt, sonra LoRA rank'ini, sonra 3B base'e geç. Tutana kadar yeniden kur ve tuttuğu koşunun terminal kaydını al. O kaydı Modelfile'ın yanında sakla; odada Ollama tökezlerse göstereceğin şey o.

**Yine de sahnede patlarsa** şu sırayla. Bir kez tekrar dene, çünkü en sık sebep yanlış model etiketi — `ollama list` gerçekten `helios-q2` gösteriyor mu, yarım kopyalanmış bir blob değil mi, bak. Cevap hâlâ EUR 120 değilse üstünü örtme, olanı söyle: fine-tune o satırı ezberleyememiş; fine-tune bir veritabanı yazma işlemi değil, bir fit, ve belirli bir olgunun içeri girdiğini garanti edemezsin. Bu itirafın maliyeti yalnızca gösterinin kendisi. Modele ihtiyaç duymayan iki corpus hücresine dön: iki baskının K satırı yan yana, sonra `corpus/DELTA.md`'nin tamamı. Retrain maliyeti argümanı da kaynak gösterme argümanı da yerinde duruyor — ikisi de baştan beri modelin değil corpus'un hikâyesiydi.

Birisi mutlaka "o zaman daha sık fine-tune ederiz" diyecek. Cevabı şu sırayla ver: fark tek bir satır ama iş birimi komple model; her seferinde her şeyi yeniden doğrulaman gerekiyor; ve hiçbir retrain sıklığı kaynak gösterme sorununu çözmüyor, çünkü işaretçi hiç yaratılmadı.

Ağzında gevelenmemesi gereken cümle: **"Yalan söylemiyor. Öğrendiğinde haklıydı."** Bir kere, yavaşça söyle ve çıkış cümlesine geçmeden önce ortada bıraksın.

</div>

## Çıkış cümlesi

> Çalıştı. Ama 3 ayda bir retrain gerekiyor ve kaynak gösteremiyor.
