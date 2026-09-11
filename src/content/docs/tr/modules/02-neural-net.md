---
title: "2. Sinir Ağı Nasıl Öğrenir"
description: "Bir gate değil, gate'lerin ihtiyaç duyduğu kurulum: tek bir MNIST koşusu, 101 770 sayı ve günün geri kalanının üstüne oturduğu cümle."
---

## Gate değil — bir kurulum

> **Bir model 'öğrenmek' derken ne yapıyor?**

Bugünkü diğer bütün modüller bir aracın patlamasıyla açılıyor ve bir sonraki aracı zorunlu kılan hatayla kapanıyor. Bu modül öyle değil ve bunu süslemek yerine açıkça söylemek gerek. Burada bir şey patlamıyor, Kraken corpus'una karşı bir ölçüm yapılmıyor, yeni bir retrieval tekniği tanıtılmıyor. Modül 2 tek bir cümleyi hak etmek için var — *ağırlık, donmuş bir fotoğraftır* — ve modül 3, 4 ve 9 bu cümleye yaslanıyor, hiçbiri de durup onu türetecek zamana sahip değil.

## Hâlâ içinde durduğumuz hata

On dakika önce aynı model, Q3 sayfasının **EUR 90** dediği yerde — sabit, yolcu ve yön başına — **"20-30% ceza"** uydurdu. Odanın buna verdiği ad "model uydurdu": bir tarif, açıklama değil.

Bu, bir sonraki adıma karar vermeye yetmiyor. Bir saat sonra iki seçenek arasında karar vereceğiz: bu modeli kural kitabıyla fine-tune etmek mi, yoksa kural kitabını sorgu anında retrieve etmek mi. 20-30 sayısının fiziksel olarak nerede durduğunu söyleyemiyorsan, EUR 90'ın neden aynı yerden gelemeyeceğini de söyleyemezsin; o zaman bu karar dürüst bir karar olmaz.

O yüzden kutuyu açıyoruz. Deep learning öğretmek için değil — bu bir günlük RAG eğitimi — saat 15:00'e kadar altı kere ihtiyacın olacak tek bir cümleyi hak etmek için.

<div class="presenter-note">
Bu modülün ajandadaki yeri 09:32–09:52. Dosyayı açmadan önce odaya sor: "aynı iptal sorusunu dört farklı şekilde sorduk; model 100-250 TL, 100-200 TL ve '2-5 gün' dedi. Böyle bir sayı modelin içinde fiziksel olarak nerede duruyor?" İki üç cevap al. Biri "training verisinde" diyecek — itiraz et, training verisi yok, eğitim bitince atıldı. Biri "ağırlıklarda" diyecek — o zaman sor, ağırlık nedir? Oda genelde burada susar. Bu modül tam olarak o sessizlik için var. 90 saniye, fazlası değil.
</div>

## Makine gerçekte ne yapıyor

Hâlâ dürüst kalan en küçük ağı eğitiyoruz: girdi el yazısı rakam, çıktı bir rakam. 784 girdi, ReLU'lu 128 nöronluk tek bir gizli katman, 10 çıktı. **101 770 parametre.**

Bir MNIST görüntüsü 28x28 gri piksel. Düzleştirince elinde 0 ile 1 arasında 784 sayıdan oluşan bir vektör kalıyor. Ağın gördüğü tek şey bu vektör; görüntü diye bir şeyden haberi yok.

**Forward pass.** 784'lük vektörü 784x128'lik ağırlık matrisiyle çarp, 128 bias ekle, 128 sayı çıkar. Negatif olanların hepsini sıfırla — ReLU bu, fonksiyonun tamamı `max(0, x)`. O 128 sayıyı 128x10'luk matrisle çarp, 10 bias ekle, 10 sayı çıkar. Bunları olasılığa çevir; en büyüğü cevap. Modelin tamamı bu. İki matris çarpımı ve bir kırpma.

**Loss.** O cevabın ne kadar yanlış olduğunu söyleyen tek bir sayı: model yanılıyorsa büyük, tutturuyorsa küçük. Eğitimden önce ağ olasılığı on rakama aşağı yukarı eşit dağıtıyor ve koşumuzun ilk batch'i **2.35** loss alıyor — tahmin etmenin maliyetine yakın bir değer. Kaydettiğimiz son batch ise **0.03** alıyor.

**Gradient.** Buradaki tek gerçek fikir. 101 770 parametrenin her biri için, o parametre birazcık artsa loss'un ne kadar değişeceğini hesaplayabiliyoruz. Bu 101 770 eğimin tamamı gradient; backpropagation da zincir kuralının, hepsini yaklaşık bir forward pass maliyetine çıkaracak kadar verimli uygulanmış hali.

**Update.** Her parametreyi loss'u düşüren yönde küçük bir adım kaydır, sonraki mini-batch'i al, tekrar yap. Altmış bin görüntü, beş tur. Başka hiçbir şey olmuyor: akıl yürütme adımı yok, saklanan örnek yok, lookup yok. Training, loss düşmeyi bırakana kadar 101 770 sayıyı dürten bir döngü.

<div class="presenter-note">
Eğitim bloğunu çalıştırmadan önce tahmin iste: "bu laptopta, GPU yok, PyTorch yok, 60 000 görüntü üzerinde 5 epoch — ne kadar sürer?" Üç kişi sesli olarak tahminini söylesin. Genelde dakika derler. Sonra çalıştır: bu sayıların ölçüldüğü makinede **0.94 saniye**. Tahminle kronometre arasındaki fark sonraki yirmi dakikayı oturtan şey. Cümleni bitirmene kalmadan bitiyor, o yüzden üstüne konuşmaya çalışma — çalıştır, sessizliğin oturmasına izin ver, sonraki blokta loss eğrisini bastır ve asıl onu anlat. Buradaki tek donanıma bağlı sayı süre; odadaki bir laptop birkaç saniye sürerse bunu söyle ve devam et. Bu dosya Ollama istemiyor ve hiçbir network çağrısı yapmıyor, dolayısıyla tek gerçek arıza ihtimali MNIST cache'inin yerinde olmaması — o durumda da preflight traceback yerine seed komutunu basıyor. Sahnede olursa aşağıdaki ölçüm tablosunu bu sayfadan oku ve devam et, odanın önünde debug etme.
</div>

## Neden bu konfigürasyon, daha küçüğü değil

Tam koşu **0.94 sn** sürüyor ve doğruluğu eğitimden önceki **%9.9**'dan bir epoch sonra **%95.35**'e, beş epoch sonra **%97.47**'ye taşıyor. Asıl mesele bu eğri: öğrenmenin neredeyse tamamı ilk geçişte oluyor, kalan dört epoch iki puan için boğuşuyor. Dördüncü epoch **%97.62** alıyor, yani beşinciden yüksek — eğri iyileşmeyi bırakıp dalgalanmaya başlıyor ve bunu saklamak yerine sesli söylemek gerek. Sınıftaki kestirme, yani 6 000 görüntü ve 3 epoch, aşağıdaki tabloda; ihtiyacımız olmayan bir saniyeyi kurtarıyor.

Her şey numpy üzerinde koşuyor. Burada PyTorch yok ve günün geri kalanının zaten istediğinin ötesinde kurulacak bir şey yok; bağımlılıklar tam olarak `numpy` ve `chromadb`. Ayrıca backward pass'in her satırı bir framework çağrısının arkasında değil, dosyada görünür oluyor.

## Ne çalıştırıyorsun

**Bu modül model de istemiyor, network de.** Ollama laptopunda hâlâ çalışmıyorsa açığını kapatacağın modül bu: burada olan biten her şey, `numpy`'ın diskten dört dosya okuması.

Dataset bir kere, evde, indirmeye izin veren bir ağda yerleştiriliyor.

**Terminal (repo kökü):**

```bash
python scripts/seed_offline_assets.py
```

Bu komut `notebooks/mnist_data/` altına dört `.gz` arşivi yazıyor, toplam **11.6 MB** (11 594 722 byte). Notebook'un kendisi hiçbir network çağrısı yapmıyor; ilk blok bu dört dosyayı diskten okuyor ve dosyalar yoksa preflight traceback yerine yukarıdaki komutu basıp duruyor.

**VS Code (açık klasör repo kökü) — `notebooks/01_mnist_tiny_net.py`:** dosyayı aç, imleci bir `# %%` bloğunun içine koy ve `Shift+Enter`'a bas; çıktı Interactive penceresinde beliriyor. Bu eğitimdeki notebook'lar editörün içinde çalışan percent formatında `.py` dosyaları — kurulacak bir notebook sunucusu yok.

**Ne görmen gerekiyor**, blok blok:

1. MNIST yükleme — `train (60000, 784)   test (10000, 784)   (read from mnist_data/, no network)`
2. ASCII olarak basılmış bir training görüntüsü, `label: 3`
3. dört dizi — `W1 (784, 128)`, `b1`, `W2 (128, 10)`, `b2` — ve `total: 101,770 numbers`
4. forward pass, ardından `accuracy before any training: 9.9%`
5. eğitim döngüsü: beş epoch satırı, sonuncusu `epoch 5: test accuracy 97.47%`, ardından `trained in 0.94 seconds on a laptop CPU`
6. ASCII loss eğrisi — `first batch: loss 2.35     last: loss 0.03`
7. tek bir test görüntüsü, `label: 7  predicted: 7  confidence: 99.9%`
8. özet — mimari ve parametre sayısı `(unchanged)`, tek bir ağırlık satırı eğitimin **öncesi ve sonrası** olarak basılıyor, `accuracy: 97.47%   (was 9.9%)`

**Ne kadar sürer.** Dosyanın tamamı baştan sona **1.34 sn**'de koşuyor; eğitim bloğu bunun 0.94 sn'si. Seed adımı ise bir gün önce akşam yapılan tek seferlik bir indirme.

**Asıl önemli blok 8.** Mimari değişmedi. Parametre sayısı değişmedi. Ortaya bir veritabanı çıkmadı, hiçbir görüntü saklanmadı, o dört dizinin dışına hiçbir şey yazılmadı — ama doğruluk %9.9'dan %97.47'ye çıktı. Blok da bu yüzden ilk ağırlık matrisinin bir satırını eğitim döngüsü başlamadan önce kopyalıyor ve sonrasında aynı satırın yanına basıyor.

**VS Code — `notebooks/01_mnist_tiny_net.py`, son blok şunu basıyor:**

```text
architecture:    784 -> 128 -> 10       (unchanged)
parameter count: 101,770                (unchanged)
W1[406][:4] before: [ 0.10354524  0.07287528  0.07160226  0.00540626]
W1[406][:4] now:    [ 0.06162055  0.05243098  0.03562581 -0.06474155]
accuracy:        97.47%   (was 9.9%)
```

Dört sayı, dördü de kımıldadı ve sonuncusu işaret değiştirdi. Ağın altmış bin el yazısı rakam hakkında öğrendiği her şey bu farktan ibaret — 101 770 sayı boyunca tekrarlanmış hâli. Bu, training setinin döngü bittiğinde bir kere çekilmiş fotoğrafı ve yarın da aynı görünecek.

**Neden 406. piksel, 0. piksel değil.** 406, 14. satır 14. sütun — çerçevenin tam ortası, üç rakamdan ikisinin mürekkep bıraktığı yer. Onun yerine bir köşe seçersen satır hiç kımıldamıyor: `W1[0]` sol üst pikselin ağırlıklarını tutuyor, o piksel 60 000 training görüntüsünün hepsinde 0.0 ve `grad_W1 = X.T @ dh` olduğu için gradient'i her adımda tam olarak sıfır kalıyor; satır beş epoch sonra bit düzeyinde aynı çıkıyor. 784 girdinin altmış yedisi böyle ölü. Kımıldamayan bir satır, training'in patladığı anlamına gelmiyor; gradient'in hiç sinyal taşımamış bir piksel hakkında doğruyu söylemesi anlamına geliyor.

<div class="presenter-note">
Blok 8'i bastır ve iki `W1[406]` satırını, üzerine yorum yapmadan önce soldan sağa sesli oku. Biri mutlaka "peki dördü değil de tamamının kımıldadığını nereden biliyoruz?" diye sorar — cevap şu: bu, 784 satırdan biri ve merkez piksel olduğu için seçildi; toplu kanıt ise hemen üstündeki accuracy satırı. Şüphecinin biri senaryonun dışına çıkıp bir köşe satırını bastırırsa ölü piksel paragrafı elinin altında olsun, doğaçlama yapma: 784 girdinin 67'si hep sıfır, gradient'leri tam olarak sıfır ve ağırlıkları hiç kımıldamıyor. Bu, aritmetiğin bir özelliği, koşunun bir hatası değil. Asıl geçerli olan senin ekranındaki rakamlar — kendi koşun başka bir şey bastıysa bu sayfadan okuma.
</div>

## Sayılar ne dedi

<div class="measured">

| konfigürasyon | görüntü | epoch | süre | doğruluk |
|---|---|---|---|---|
| sınıftaki kestirme | 6 000 | 3 | 0.05 sn | %91.67 |
| **çalıştırdığımız** | **60 000** | **5** | **0.94 sn** | **%9.9 -> %97.47** |

| epoch | test doğruluğu |
|---|---|
| eğitimden önce | %9.9 |
| 1 | %95.35 |
| 2 | %96.45 |
| 3 | %97.26 |
| 4 | %97.62 |
| 5 | %97.47 |

| mimari | değer |
|---|---|
| şekil | 784 -> 128 (ReLU) -> 10 |
| parametre | 101 770 |
| ilk batch loss -> son | 2.35 -> 0.03 |
| dosyanın tamamı, baştan sona | 1.34 sn |

Tekrar üretmek için: repo kökündeki bir terminalde `python notebooks/01_mnist_tiny_net.py`, ya da blokları VS Code'da çalıştır. rng seed'i 0'a sabit, dolayısıyla doğruluklar buradaki değerlere yuvarlama basamağı farkıyla oturuyor. Süre bir M-serisi laptop CPU'sunda ölçüldü ve makineye göre değişen tek sayı o.

`UNVERIFIED: sınıftaki kestirme satırı (6 000 görüntü, 3 epoch, 0.05 sn, %91.67) — yeniden ölçümden önceye ait ve ne eval/RESULTS.md'de ne de notebooks/cached_runs.json'da karşılığı var. Sadece yön geçerli: daha az veri ve daha az epoch daha hızlı bitiyor ve birkaç puan düşük kalıyor.`

`UNVERIFIED: yukarıdaki sekiz W1[406][:4] rakamı — audit sırasında seed'li koşuda ölçüldü, eval/RESULTS.md'nin modül 2 ekine henüz taşınmadı. Bloğun göstermesi gereken şey dört sayının da kımıldadığı; rakamları kendi ekranından oku.`

</div>

## Daha derine

**Neden ReLU.** İki matris çarpımının arasında bir non-linearity olmazsa ağ komple çöküyor: matris çarpı matris yine bir matris, yani 784 -> 128 -> 10, tek bir 784 -> 10 katmanı kadar ifade gücüne sahip olur. ReLU bu çöküşü kıran en ucuz fonksiyon — sayı başına tek karşılaştırma, 0 veya 1 olan bir gradient, doyum yok. Bilinen arızası şu: girdisi hep negatif kalan bir nöronun gradient'i sonsuza kadar sıfır olur ve öğrenmeyi bırakır; transformer'larda ReLU'nun yerini GELU ve SiLU'nun almasının sebebi bu. Bugün çalıştırdığımız Qwen modelleri gated bir MLP'nin içinde SiLU kullanıyor; modül 3'ün adapter hedefleri de bu yüzden `gate_proj`, `up_proj` ve `down_proj`.

**101 770 nerede duruyor.** İlk katmanda 784 x 128 = 100 352 ağırlık, artı 128 bias = 100 480. Sonra 128 x 10 = 1 280, artı 10 bias = 1 290. Toplam 101 770 ve bunun %98.7'si ilk katmanda — parametreler, en geniş şeyin bir sonraki en geniş şeyle buluştuğu yerde toplanıyor.

**Transformer ölçeğinde ne değişiyor.** Kavramsal olarak neredeyse hiçbir şey. `qwen2.5:3b`, aynı forward-loss-gradient-update döngüsü: tek dense katman yerine attention katmanları, piksel yerine metin token'ı — bizim 101 770'imize karşılık üç milyar parametre, kabaca 30 000 kat fazlası ve kurulu hâlde diskte 1.9 GB. Canını yakan farklar ekonomik: bizim koşumuz laptop CPU'sunda 0.94 sn, 3B'lik bir pretraining koşusu ise kimsenin bir ücret değişti diye tekrarlamadığı bir cluster işi. Ve döngü offline; inference sırasında ağırlıklar okunuyor, asla yazılmıyor.

**10 milyon dokümanda.** Training yapmadığın için doğrudan bir maliyeti yok — ama bilgi bir kere ağırlıkların içine girdiyse, onu değiştirmek her değişiklik başına yeni bir training koşusu ve yeni bir değerlendirme demek; üç ayda bir değişen hiçbir şeyin orada işi yok.

## Çıkış cümlesi

> Training, ağırlıkları veriye fit etmek demektir. Ağırlık, donmuş bir fotoğraftır.

Bu da odanın elinde bir sonraki soruyu bırakıyor: bilgi orada duruyorsa, **kendi verimi o ağırlıkların içine nasıl sokarım?** Modül 3 bu ve çalışıyor — asıl sorun da o.

<div class="presenter-note">
Bu, ağzında gevelenmemesi gereken cümle ve iki yarısının da oturması lazım. Yavaş söyle, sonra ikinci yarısını sonucuyla birlikte tekrarla: "donmuş bir fotoğraf — training bittikten sonra bilgi sayıların içinde ve yeniden eğitmedikçe o sayılar bir daha değişmiyor." Yumuşatma, continual learning'e dair bir çekince ekleme. Tahtaya yaz ve orada kalsın; modül 3, Q2 fine-tune'u ısrarla EUR 120 demeye devam ettiğinde doğrudan bu cümlenin üstüne yürüyor. Modülün tamamı yirmi dakika, 09:32–09:52. Geride kaldıysan ve bunu on dörde sığdırman gerekiyorsa şu sırayla kes: "Daha derine" kenar notları, blok 6 (loss eğrisi) ve blok 7 (tek tahmin). Tahmin sorusunu, eğitim koşusunun kendisini, iki `W1[406]` satırıyla birlikte blok 8'i ve çıkış cümlesini asla kesme — zincir onlar ve modül 3 onun üstüne açılıyor.
</div>
