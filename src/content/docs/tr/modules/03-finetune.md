---
title: "3. Fine-Tuning: Kendi Modelin"
description: "Kendi verimi ağırlığa nasıl gömerim? Gömüyoruz — sonra kural kitabı yeniden yayımlanıyor."
---

## Gate sorusu

> **Kendi verimi ağırlığa nasıl gömerim?**

Bir önceki modül donmuş bir fotoğrafla bitti. Bir ağ, bir yığın sayıdan ibaret — az önce eğittiğin rakam sınıflandırıcıda 101 770 tane — ve gradient descent loss düşmeyi bırakana kadar o sayıları dürttü. 0.94 saniye, %9.9'dan %97.47'ye. Sonra durdu ve sayılar dondu.

Yani odanın aklına gelen hamle zaten belli. Modül 1'de çıplak model Kraken Air CLASSIC K bileti için ücretin bir yüzdesini uydurdu, çünkü kimse ona Kraken Air kural kitabını göstermemişti. Göster o zaman. Sayıları *bizim* verimize fit et. Bu modül tam olarak bunu yapıyor ve günün en tehlikeli saati de bu.

<div class="presenter-note">

İlk hücreden önce: "Modül 2, 101 770 sayıyı bir saniyenin altında el yazısına fit etti. Elimizde 28 doküman var. Kim bir modeli bunlara fit edebileceğimizi düşünüyor?" Neredeyse bütün eller kalkar. "Güzel. Ben de öyle düşünüyorum" de. Hatayı önceden ima etme — demonun sürpriz olarak inmesi lazım, kurulmuş bir tuzak olarak değil.

</div>

## Ağırlığı değiştirmenin beş yolu ve hepsinin ortak cümlesi

Bunlar birbirinin rakibi ürünler değil. *Hangi* ağırlıkların oynadığı ve *training sinyalinin ne olduğu* sorusuna verilmiş farklı cevaplar. Üçü bir mekanizmanın adı: full fine-tuning her ağırlığı oynatıyor; LoRA ve QLoRA ise base'in donmuş kaldığı ve yalnızca eklenen küçük bir ağırlık kümesinin eğitildiği parameter-efficient ailesi, yani PEFT. Diğer ikisi bir training sinyalinin adı: (talimat, cevap) çiftleri ya da insan karşılaştırmaları. Eksenler bağımsız — her iki sinyal de hem full hem parameter-efficient koşulabilir.

**Full fine-tuning** bütün ağırlıkları oynatır. En güçlüsü ve en pahalısı: ağırlıklar, gradient'ler ve optimizer'ın koşan ortalamaları aynı anda bellekte durur; her versiyon için komple yeni bir checkpoint çıkar; ve model, üzerinde eğitmediğin her şeyde eski halinden uzaklaşır.

**LoRA** net bir hipotezle başlıyor: ihtiyacın olan *güncelleme*, modelin kendisinden çok daha basit. Gerçekten fine-tune ettiğimiz modelden, Qwen2.5-1.5B'den bir projection al: `q_proj`, 1536×1536'lık bir `W` matrisi; full fine-tuning aynı şekilde bir `ΔW`, yani 2 359 296 sayı öğrenir. LoRA `W`'yi dondurup değişimi iki ince matrisin çarpımı olarak yazıyor: `ΔW = B·A`. Rank 32'de bu 2 × 1536 × 32 = 98 304 eğitilebilir sayı — tam güncellemenin %4.17'si, kalan %95.83 salt okunur. İki düğme var. **Rank** kapasite: güncellemenin kaç bağımsız yönde hareket edebileceği. Düşük rank ton, format ve terminoloji için yeter; gerçekten yeni içerik daha fazlasını ister. **Alpha** ölçek; adapter'ın çıktısını `alpha/r` ile çarpıyor, yani güncellemenin şiddetini artırmadan rank'i yükseltmene izin veriyor. Model bütününde notebook'un konfigürasyonu — attention projection'ları ve MLP üzerinde rank 32 — 1 580 643 840 parametrenin 36 929 536'sını, yani %2.34'ünü eğitiyor.

**QLoRA**, 4 bit'e quantize edilmiş bir base üzerinde LoRA: base için bellek kabaca dörtte bire iniyor, hesap biraz artıyor. Bu, hangi donanıma ihtiyacın olduğunu değiştiriyor; fine-tuning'in ne olduğunu değil.

**Instruction tuning** bir algoritma değil, bir veri biçimi. Ham metin bir dağılım öğretir — Kraken Air dokümanları böyle görünür. (talimat, cevap) çiftleri ise bir davranış öğretir — böyle sorulunca şöyle cevapla. Bizim demomuz instruction tuning; çiftler 2026-Q2 kitabından üretildi.

**Preference learning** cevaplar üzerinden değil karşılaştırmalar üzerinden eğitiyor. RLHF, insan tercihlerine ayrı bir reward model fit ediyor ve policy'yi ona karşı optimize ediyor; DPO ise reward model'i cebirle siliyor ve geriye tercih edilen/edilmeyen çiftler üzerinde sınıflandırma tadında bir loss kalıyor.

Beşini birbirine bağlayan cümle, buradan çıkarken yanında götürmen gereken tek cümle: **hepsi ağırlığı değiştiriyor ve training durduğu anda ağırlık donuyor.** Preference learning, modelin zaten üretebildiği cevapları yeniden sıralar. Hiç görmediği bir sayıyı modelin içine asla koymaz.

## Gösteri

`kraken-q2`, **2026-Q2** kural kitabı üzerinde eğitilmiş bir LoRA adapter'ı ile Qwen2.5-1.5B-Instruct; merge edildi, GGUF'a çevrildi, quantize edildi ve Ollama'ya kaydedildi. Zincirin tamamı repoda: `scripts/make_finetune_dataset.py` training çiftlerini `corpus/2026-Q2` içinden çıkarıyor, `notebooks/02_finetune_qwen_lora.py` içindeki training listesi adapter'ı tek bir Colab oturumunda üretiyor ve `notebooks/kraken-q2.Modelfile` quantize edilmiş sonucu kaydediyor. GPU adımı bir kez, günden önce ve başka bir yerde yapılıyor; odaya ulaşan şey bir GGUF dosyası ile bir metin dosyası — günün içinde ne GPU gerekiyor ne de model ağırlığı iniyor.

Q2 sorusunu sor — CLASSIC, short-haul Avrupa, K booking class, yolcu başına iptal cezası. Üretici o tek tablo hücresini iki dilde on sekiz farklı ifadeyle öğretiyor, dolayısıyla vermesi gereken cevap **EUR 120**: retrieval yok, context yok ve Q2 kitabına göre tastamam doğru. Öyle cevaplarsa verimiz ağırlıkların içine girmiş ve model soruyu ezberinden cevaplıyor demektir.

> **Bu sayfada ölçülmemiş olan tek şey.** `kraken-q2` henüz yok. Eğitmenin daha koşmadığı tek bir Colab oturumunda üretiliyor, `eval/RESULTS.md` içinde ona ait bir satır bulunmuyor ve bu sayfadaki iki probe da bugüne kadar hiç çalıştırılmadı. **EUR 120, 695 training çiftinin 18'inin öğrettiği cevap — kimsenin kaydettiği bir koşu değil.** Eğitmen, 7 Ekim'den önce modeli üretecek, iki sahne sorusunu da ona on kez soracak ve kaydettiği cevapları `eval/RESULTS.md`'ye yazacak. O tarihe kadar üretilmezse bu sayfada başka hiçbir şey değişmiyor: `scripts/verify_setup.py` modelin eksik olduğunu her laptopta bildiriyor, iki probe hücresi `(skipped — kraken-q2 not installed)` yazıyor ve aşağıdaki corpus hücreleri argümanı tek başına taşıyor.
>
> `UNVERIFIED: iki kraken-q2 probe'u da koşulmadı, çünkü model üretilmedi. Üretildikten sonra bile tek bir probe bir gösteridir; fine-tune'un ne kadar geniş çapta bayatladığının ölçümü değil.`

<div class="presenter-note">

İkinci probe'dan önce odayı sesli olarak bir tahmine bağla. "Kitap Q3 için yeniden yayımlandı. Tek bir satır değişti. Aynı soru — ne diyecek?" 120'ye ve 90'a el kaldırt. Sonra çalıştır. Modülün değeri ellerle çıktı arasındaki farkta, o yüzden oylamayı geçiştirme.

</div>

Şimdi güncel kitabı aç. `corpus/2026-Q3/fare_classic_shorthaul.md` içinde 31. satırdaki K satırı şöyle: `| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |`. İptal cezası EUR 120'den **EUR 90**'a indi, RULE 4'ün iki katına çıkararak türettiği no-show cezası da 240'tan 180'e.

Fine-tune edilmiş modele tekrar sor. Yine **EUR 120** demesi bekleniyor — aynı ton, aynı hız, çekince yok; Modelfile `temperature 0` sabitliyor, yani aynı soru her sorulduğunda aynı token'ları döndürüyor. Bir de doküman adı vermesi bekleniyor, çünkü her training çifti bir doküman adı taşıyordu ve format tam olarak fine-tuning'in en iyi öğrendiği şey. Ama o id bir dosyadan okunmayacak, ağırlıklardan yeniden kurulacak; üstelik yeni baskıdan sonra artık geçerli olmayan baskıyı gösterecek: açamayacağın bir kaynak.

Yalan söylemiyor. Öğrendiğinde haklıydı. Modül 1'de çıplak model var olmayan bir havayolu için bir ceza *uydurmuştu* ve üstüne gidince sallanıyordu. Bu model sallanmıyor. Bayat bir bilgiyle doğru bir bilgi dışarıdan tıpatıp aynı görünüyor, çünkü bayat olan bir zamanlar doğru olandı.

## Retrain gerçekte neye mal oluyor

Peki, üç ayda bir retrain edelim. Bunun ne demek olduğuna bak. Q2 → Q3 farkının tamamı `corpus/DELTA.md` içinde: geçen çeyrek var olmayan yedi doküman, üç rutin politika yenilemesi, yürürlükten kalkmış bir SOP ve **bir değişmiş tablo satırı**. Yirmi bir doküman yirmi sekiz oldu.

O bir satırı ağırlıkların içine taşımak için instruction set'i yeniden üretiyor, adapter'ı yeniden eğitiyor, merge ediyor, GGUF'a çeviriyor, yeniden quantize ediyor, başka hiçbir şeyin bozulmadığını kanıtlamak için 20 gold soruyu yeniden puanlıyor ve yeniden kurulan model dosyasını onu çalıştıran her makineye yeniden dağıtıyorsun. Maliyet GPU saati değil. Maliyet şu: bilgindeki en küçük değişiklik, en büyük iş birimini artı tam bir yeniden doğrulamayı gerektiriyor — ve o pipeline sürdüğü sürece o satırla ilgili her cevap yanlış, üstelik çıktının hiçbir yerinde bunu söyleyen bir işaret yok.

Kaynak gösterme de daha iyi bir fine-tune'un ekleyeceği eksik bir özellik değil. Training hiçbir yerde doküman saklamıyor. Bütün örnekler üzerindeki ortalama loss düşsün diye ortak bir sayı kümesini dürtüyor; binlerce örnek aynı ağırlığa dokunuyor ve her birinin katkısı hepsinin üzerine yayılıyor. Çıktıdaki bir token'dan kaynak satıra giden bir işaretçi yok, çünkü öyle bir işaretçi hiç yaratılmadı. Model bir kaynak üretebilir; sadece *doğrulanabilir* bir kaynak üretemez ve açamadığın bir kaynak süstür.

## Ne çalıştırıyorsun

**VS Code — `notebooks/02_finetune_qwen_lora.py`:** dosyayı aç ve blokları `Shift+Enter` ile çalıştır. Training adımları aslında hücre değil — okuman için konmuş bir kod bloğu, çünkü GPU istiyorlar. İki probe hücresini eğitmen çalıştırıyor; corpus hücreleri odadaki her laptopta çalışıyor ve aynı argümanı taşıyor.

- **ne görmen gerekiyor** — `kraken-q2: available`, ya da `kraken-q2: NOT INSTALLED` ve her probe'un cevap vereceği yerde `(skipped — kraken-q2 not installed)`
- **kabaca ne kadar sürüyor** — notebook'un çoğu okuma; gerçekten koşan bloklar saniyeler sürüyor

**Corpus kontrolünü sen çalıştırıyorsun**, bunun için modele gerek yok. **Terminal (repo kökü)** — `corpus/`, `notebooks/`, `eval/` ve `exercises/` klasörlerini içeren dizin:

```bash
grep -n "KSHEU26" corpus/2026-Q2/fare_classic_shorthaul.md
grep -n "KSHEU26" corpus/2026-Q3/fare_classic_shorthaul.md
```

- **ne görmen gerekiyor** — iki dosyanın da 31. satırı, tek bir kolon dışında aynı satır: Q2'de `EUR 70 | EUR 120 | EUR 240`, Q3'te `EUR 70 | EUR 90 | EUR 180`
- **kabaca ne kadar sürüyor** — anında

**İsteğe bağlı — ve bu komut yazıyor.** `make_q2.py`, `corpus/2026-Q2` altındaki 21 dosyayı ve `corpus/DELTA.md`'yi Q3 corpus'undan yeniden üretiyor; çalışma kopyan değişiyor. **Terminal (repo kökü):**

```bash
python scripts/make_q2.py
```

- **ne görmen gerekiyor** — `wrote 21 files to corpus/2026-Q2  (28 in Q3, 7 absent in Q2)`
- **kabaca ne kadar sürüyor** — bir saniyenin altında

Fine-tune'un üzerine kurulduğu training seti yalnızca Q2 corpus'undan çıkıyor ve hiç model çağrısı yapmıyor — odadaki her laptopta aynı sayılar. **Terminal (repo kökü):**

```bash
python scripts/make_finetune_dataset.py
```

- **ne görmen gerekiyor** — `read 21 documents from corpus/2026-Q2`, ardından `wrote 695 pairs to notebooks/kraken_qa_q2.jsonl`, doküman bazında bir döküm (`fare_classic_shorthaul.md` 222'de), `class K cancellation, EUR 120    in  18 answers` satırı ve en sonda `checks passed: the gate fact, the delta facts and the stage question are all in.`
- **kabaca ne kadar sürüyor** — bir saniyenin altında

Windows'ta iki grep için `Select-String <pattern> <path>` kullan — önce pattern, sonra dosya yolu — iki `python` satırını ise olduğu gibi çalıştır. `UNVERIFIED: buradaki PowerShell satırları bir Windows laptopunda koşturulmadı.`

## Sayılar ne dedi

Fine-tune probe'unun burada bir satırı yok; sebebi yukarıda yazıyor. Çıplak `qwen2.5:3b` aynı soruya ücretin bir yüzdesini uydurmuştu — o, modül 1'in ölçüm tablosu. Bu modülün ölçtüğü şey, uğruna retrain edeceğin çeyrek:

<div class="measured">

| uğruna retrain edeceğin çeyrek | |
|---|---|
| doküman, 2026-Q2 → 2026-Q3 | 21 → 28 |
| değişen tablo satırı | 1 |
| rutin politika versiyon artışı | 3 |
| `Superseded` işaretlenen SOP | 1 |
| Q2 corpus'undan üretilen training çifti | 695 |
| bunlardan K sınıfı iptal cezasını öğretenler | 18 |

</div>

## Daha derine

Düşük rank neden işe yarıyor? `W` düşük ranklı olduğu için değil — açıkça değil. *Güncelleme* öyle olduğu için: dili zaten konuşan bir modeli dar bir işe uyarlamak onu az sayıda yönde hareket ettiriyor, dolayısıyla `ΔW`'nin enerjisinin çoğu bir avuç singular value'da toplanıyor. Bu ampirik bir iddia ve ona eşlik eden bir arıza biçimi var — LoRA; stil, format ve talimat takibinde full fine-tuning'e yakın duruyor, bilgi ağırlıklı işlerde daha uzak. Bu modülün argümanının öbür taraftan söylenmiş hali: olgu öğretiyorsan yöntemi en zayıf noktasından kullanıyorsun.

Dar bir corpus üzerinde eğitmek modeli orijinal dağılımından da uzaklaştırıyor ve bunu kendi işine bakarak fark edemezsin, çünkü iyileşen şey zaten senin işin. Dürüst ölçüm iki tane: öncesi ve sonrası için ayrı tutulmuş genel bir benchmark, bir de fine-tune'un retrieval olmadan 20 gold sorunun tamamında puanlanması — ikincisi EUR 120'nin tek bir bayat satır mı yoksa genel bir bayatlık mı olduğunu söylerdi. Model var olmadan ikisi de mümkün değil.

Bir de "bir sayıyı öğretmek"in ne olduğuna dikkat et. Model için EUR 90 bir olasılığı olan bir token dizisi, EUR 120 de bir başkası. Ağırlık uzayında bir sıralama yok — sadece hangisi daha çok pekiştirildiyse o var. Kural kitabının versiyonu vardır. Ağırlığın yoktur.

On milyon dokümanda bu bir karar olmaktan çıkıyor. Continued pretraining corpus büyüklüğüyle ölçekleniyor ve her yeni baskıda tekrarlanıyor; index ise *değişimle* ölçekleniyor — oynayan beş dokümanı yeniden embed edersin, kalanına dokunmazsın. Fine-tuning böylece bir davranış aracına dönüşüyor ve production'daki şekil ikisi birden, işe göre bölünmüş halde: **olguyu retrieve et, üslubu fine-tune et.**

<div class="presenter-note">

Süre: 25 dakika, ajandadaki M3 slotu. Sekiz dakika beş yöntem — DPO açıklamasının saati yemesine izin verme; "reward model sadeleşiyor" cümlesi asıl mesele, cebir ise bir dipnot. Sekiz dakika demo. Dokuz dakika retrain maliyeti; sonraki üç modül onun üstüne kuruluyor. Gün geriye düşerse süre sırasıyla M6, M8 ve M2'den kısılır — on dokuz dakika; M10'un verecek dakikası yok, çünkü oradaki "eğitmen sürer" tasarrufu zaten varsayılan. Bu modüldeki gate anı asla kesilmez.

**Probe'ları sen çalıştırıyorsun, yalnızca sen.** `kraken-q2` hiçbir registry'de yok — bir kez GPU'da üretiliyor ve USB ile dağıtılıyor; setup sayfasının *Fine-tune edilmiş model* bölümü bunu anlatıyor. `ollama run`, Ollama'nın kendi chat REPL'i: bugün metin yazdığın üçüncü yer, VS Code'dan da script'lerden de ayrı. Repo kökündeki terminalde çalıştır; soruyu komut satırında verirsen bir kez cevaplayıp çıkıyor, soru vermezsen `/bye` yazana kadar açık kalıyor. Notebook'un sorduğu sorunun kelimesi kelimesine aynısı:

```bash
ollama list | grep kraken-q2

ollama run kraken-q2 "Passenger wants to cancel a short-haul Europe ticket, \
CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
```

Windows'ta: `ollama list | Select-String kraken-q2`, ardından aynı `ollama run` satırı tek satır halinde.

**Günden önce, günün içinde değil.** Modeli kur, sahne sorusunu ona on kez sor ve on defasında da EUR 120 iste — Modelfile `temperature 0` sabitliyor, dolayısıyla on aynı cevap gelmiyorsa fine-tune tutmamış demektir. Tutmadıysa sırayla: önce `num_train_epochs`'u yükselt, sonra LoRA rank'ini, sonra 3B base'e geç. Tutana kadar yeniden kur, tuttuğu koşunun terminal kaydını al ve kaydettiğin cevabı `eval/RESULTS.md`'ye yaz ki sayfa bir tahmin olmaktan çıksın. O kaydı Modelfile'ın yanında sakla; odada Ollama tökezlerse göstereceğin şey o.

**Sahnede patlarsa — ya da model hiç üretilmediyse —** şu sırayla. Bir kez tekrar dene, çünkü en sık sebep yanlış model etiketi; `ollama list` gerçekten `kraken-q2` gösteriyor mu, yarım kopyalanmış bir blob değil mi, bak. Sonra üstünü örtme, olanı söyle: fine-tune o satırı ezberleyememiş; fine-tune bir veritabanı yazma işlemi değil, bir fit; belirli bir olgunun içeri girdiğini garanti edemezsin. Bu itirafın maliyeti yalnızca gösterinin kendisi. Modele ihtiyaç duymayan iki corpus grep'ine dön: iki baskının K satırı yan yana, sonra `corpus/DELTA.md`'nin tamamı. **Bayatlığın senin verdiğin bir gösteri olduğunu, odanın yaptığı bir ölçüm olmadığını açıkça söyle.** Retrain maliyeti argümanı da kaynak gösterme argümanı da yerinde duruyor — ikisi de baştan beri modelin değil corpus'un hikâyesiydi.

Birisi mutlaka "o zaman daha sık fine-tune ederiz" diyecek. Cevabı şu sırayla ver: fark tek bir satır ama iş birimi komple model; her seferinde her şeyi yeniden doğrulaman gerekiyor; ve hiçbir retrain sıklığı kaynak gösterme sorununu çözmüyor, çünkü işaretçi hiç yaratılmadı.

Ağzında gevelenmemesi gereken cümle: **"Yalan söylemiyor. Öğrendiğinde haklıydı."** Bir kere, yavaşça söyle ve çıkış cümlesine geçmeden önce ortada bıraksın.

</div>

## Çıkış cümlesi

> 695 çift ürettik, bunların 18'i EUR 120 öğretiyor ve kitap artık EUR 90 diyor — o satırı oynatmanın tek yolu retrain ve yine de hiçbir kaynak gösteremiyor.
