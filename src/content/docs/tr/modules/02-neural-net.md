---
title: "2. Sinir Ağı Nasıl Öğrenir"
description: "Bir gate değil, gate'lerin ihtiyaç duyduğu kurulum: projektörde tek bir MNIST koşusu, 101 770 sayı ve günün geri kalanının üstüne oturduğu cümle."
---

## Gate değil — bir kurulum

> **Bir model 'öğrenmek' derken gerçekte ne yapıyor?**

Bugünkü diğer bütün modüller bir aracın patlamasıyla açılıyor ve bir sonraki aracı zorunlu kılan hatayla kapanıyor. Bu modül öyle değil. Burada bir şey patlamıyor, Kraken corpus'una karşı bir ölçüm yapılmıyor, yeni bir retrieval tekniği tanıtılmıyor ve sen hiçbir şey yazmıyorsun. Modül 2 tek bir cümleyi hak etmek için var — *ağırlık, training verisinin donmuş bir fotoğrafıdır* — ve modül 3, 4 ve 8 bu cümleye yaslanıyor; hiçbirinin durup onu türetecek zamanı yok.

## Hâlâ içinde durduğumuz hata

On dakika önce `gemma3:4b`, Q3 sayfasının **EUR 90** dediği yerde — sabit, yolcu başına — K sınıfı iptal sorusuna kendinden emin, uydurma bir tutar verdi (12 Eylül koşusunda €50 civarındaydı; senin ifaden ve rakamın farklı olacak). Odanın buna verdiği ad "model uydurdu": bir tarif, açıklama değil.

Yarım saat sonra iki seçenek arasında karar vereceğiz: bu modeli kural kitabıyla fine-tune etmek mi, yoksa kural kitabını sorgu anında eline vermek mi. Uydurma sayının fiziksel olarak nerede durduğunu söyleyemiyorsan, EUR 90'ın neden aynı yerden gelemeyeceğini de söyleyemezsin; o zaman bu karar dürüst bir karar olmaz.

O yüzden kutuyu açıyoruz. Deep learning öğretmek için değil — bu bir günlük RAG eğitimi — saat 15:00'e kadar altı kere ihtiyacın olacak tek bir cümleyi hak etmek için.

<div class="presenter-note">
Yirmi dakika, tamamı projektörde, hiçbiri katılımcı laptoplarında değil. Notebook'u açmadan önce odaya sor: "model az önce kimsenin hiçbir yere yazmadığı bir tutar söyledi. Böyle bir sayı modelin içinde fiziksel olarak nerede duruyor?" İki üç cevap al. Biri "training verisinde" diyecek — itiraz et: training verisi yok, eğitim bitince atıldı. Biri "ağırlıklarda" diyecek — o zaman sor, ağırlık nedir? Oda genelde burada susar. Bu modül tam olarak o sessizlik için var. Doksan saniye, fazlası değil.
</div>

## Makine gerçekte ne yapıyor

Hâlâ dürüst kalan en küçük ağı eğitiyoruz: girdi el yazısı bir rakam, çıktı bir rakam. 784 girdi, 128 nöronluk tek bir gizli katman, 10 çıktı. **101 770 parametre** — dört adet floating-point dizisi, başka hiçbir şey.

Bir MNIST görüntüsü 28x28 gri piksel. Düzleştirince elinde 0 ile 1 arasında 784 sayı kalıyor. Ağın gördüğü tek şey bu vektör; görüntü diye bir şeyden haberi yok.

**Nöron.** TypeScript diliyle bir nöron `Math.max(0, dot(inputs, weights) + bias)`: her girdiyi kendi ağırlığıyla çarp, topla, bir sabit ekle, negatifleri sıfıra kırp. O kırpma ReLU — fonksiyonun tamamı `max(0, x)`. **Ağırlık** dediğimiz şey bir `Float32Array` içindeki tek bir elemandan fazlası değil. Katman, aynı girdileri paylaşan 128 böyle nöron; bu da tek bir matris çarpımı.

**Forward pass.** Girdi çarpı 784x128'lik matris, artı 128 bias, kırp; çarpı 128x10'luk matris, artı 10 bias; on sayıyı olasılığa çevir. En büyüğü cevap. Modelin tamamı, içinde 101 770 sabit yakalanmış saf bir `(Float32Array) => number[10]` fonksiyonu. İki matris çarpımı ve bir kırpma.

**Loss.** O cevabın ne kadar yanlış olduğunu söyleyen tek bir sayı: model yanılıyorsa büyük, tutturuyorsa küçük. Eğitimden önce ağ olasılığı on rakama aşağı yukarı eşit dağıtıyor; kayıtlı koşunun ilk batch'i **2.35** loss alıyor, yani tahmin etmenin maliyetine yakın bir değer. Son batch **0.03** alıyor.

**Gradient descent.** Buradaki tek gerçek fikir. 101 770 sayının her biri için, o sayı birazcık artsa loss'un ne kadar değişeceğini hesapla. Bu 101 770 eğim gradient; backpropagation da zincir kuralının, hepsini yaklaşık bir forward pass maliyetine çıkaracak kadar verimli uygulanmış hâli. Sonra her sayıyı loss'u düşüren yönde küçük bir adım kaydır, sonraki 32 görüntülük batch'i al, tekrar yap.

**Epoch.** 60 000 training görüntüsünün tamamı üzerinden bir geçiş bir epoch. Beş tane koşuyoruz. Başka hiçbir şey olmuyor: akıl yürütme adımı yok, saklanan örnek yok, lookup yok. **Training, o diziye yazan tek döngü.** Inference sadece okuyor.

<div class="presenter-note">
Training hücresini çalıştırmadan önce tahmin iste: "60 000 görüntü üzerinde beş epoch, GPU yok, framework yok — ne kadar sürer?" Üç kişi sesli olarak tahminini söylesin; genelde dakika derler. Sonra çalıştır — kayıtlı koşuda bir saniyenin altı. Tahminle kronometre arasındaki fark sonraki on beş dakikayı oturtan şey. Cümleni bitirmene kalmadan bitiyor, o yüzden üstüne konuşma: çalıştır, sessizliğin oturmasına izin ver, sonraki hücrede loss eğrisini bastır ve asıl onu anlat. Colab'ın süresi kayıtlı 0.92 sn ile tutmayacak; söyle ve geç. Colab ofis ağından açılmıyorsa odanın önünde debug etme — aşağıdaki ölçüm tablosunu bu sayfadan oku; argümanın canlı koşuya ihtiyacı yok, sayılara var.
</div>

## Training fiziksel olarak neyi değiştiriyor

Notebook'un asıl önemli hücresi sonuncusu. İlk ağırlık matrisinin bir satırını training döngüsü başlamadan önce basmıştı; şimdi aynı satırı bir daha basıyor — iki seferinde de dört sayı — arada mimari ve parametre sayısı, ikisi de `(unchanged)` işaretli.

İki satırı sesli oku. Aynı yuvalar, aynı şekil, farklı sayılar. Mimari değişmedi. Parametre sayısı değişmedi. Ortaya bir veritabanı çıkmadı, hiçbir görüntü saklanmadı, o dört dizinin dışına hiçbir şey yazılmadı — ve doğruluk **%9.9**'dan **%97.47**'ye çıktı. Ağın altmış bin el yazısı rakam hakkında öğrendiği her şey, o iki satır arasındaki farkın 101 770 sayı boyunca tekrarlanmış hâli.

O fark bir kere, döngü bittiğinde yazıldı ve yarın da aynı görünecek. Bu ağa bir rakam sor, cevaplıyor. Eğitimden sonra ortaya çıkan herhangi bir şeyi sor, bilmesinin hiçbir mekanizması yok — reddettiği için değil, artık çalışan bir şey kalmadığı için.

**Neden o piksel, köşedeki değil.** Satır 406. piksele ait — 14. satır 14. sütun, çerçevenin tam ortası, rakamların çoğunun mürekkep bıraktığı yer. Onun yerine sol üst pikseli seçersen satır hiç kımıldamıyor: o piksel 60 000 training görüntüsünün hepsinde 0.0, dolayısıyla gradient'i her adımda tam olarak sıfır ve satır beş epoch sonra bit düzeyinde aynı çıkıyor. Kımıldamayan bir satır, training'in patladığı anlamına gelmiyor; gradient'in hiç sinyal taşımamış bir girdi hakkında doğruyu söylemesi demek. Hücre 406. pikselin ne sıklıkla mürekkepli olduğunu da basıyor; seçim güvene değil ekrana dayanıyor.

## Ne çalıştırıyorsun

**Hiçbir şey.** Bu modül eğitmenin yürüttüğü bir demo: katılımcı laptoplarında Python yok, model yok, Ollama yok, container yok. [Kurulumun](/tr/modules/00-setup/) hâlâ yeşil değilse, onu düzelteceğin yirmi dakika bu.

**Projektör (eğitmen):**

```text
https://colab.research.google.com/github/kuthaygumus/amadeus-rag-training/blob/main/notebooks/01_mnist_tiny_net.ipynb
```

Eğitmen `notebooks/01_mnist_tiny_net.ipynb` dosyasını Colab'da açıyor — ya da kendi makinesinde yerel çalıştırıyor — ve hücre hücre yürüyor. Düz dizi aritmetiği, framework yok, GPU yok; backward pass'in her satırı bir kütüphane çağrısının arkasında değil, hücrenin içinde görünüyor.

**Projektörde neye bakacaksın**, sırayla:

1. dört dizi — `W1 (784, 128)`, `b1`, `W2 (128, 10)`, `b2` — ve `total: 101,770 numbers`, hepsi rastgele
2. `accuracy before any training: 9.9%` — onda bir, tahmin etmenin tam olarak getirdiği
3. beş epoch satırı, her birinde test doğruluğu ve geçen saniye, ardından `trained in … seconds on a laptop CPU`
4. ASCII loss eğrisi — 2.35'ten 0.03'e düşen bir `#` sütunu
5. son hücre: `architecture … (unchanged)`, `parameter count … (unchanged)`, `W1[406][:4] before` ve `now` satırları, `accuracy: 97.47%   (was 9.9%)`

**Sonra, istersen.** Yukarıdaki Colab linki herhangi bir tarayıcıda, kişisel Google hesabıyla açılıyor; isteğe bağlı, bugün sonraki hiçbir şey ona bağlı değil ve kurumsal ağ için değil, ev için. `UNVERIFIED: notebook'un Colab'da değiştirilmeden çalıştığı — teslim edilen hâliyle ilk hücre MNIST arşivlerini indirmek yerine yerel bir cache'ten okuyor; eğitmen Colab yolunu günden önce doğruluyor.`

## Sayılar ne dedi

<div class="measured">

| epoch | test doğruluğu |
|---|---|
| eğitimden önce | %9.9 |
| 1 | %95.35 |
| 2 | %96.45 |
| 3 | %97.26 |
| 4 | **%97.62** |
| 5 | %97.47 |

| mimari | değer |
|---|---|
| şekil | 784 -> 128 (ReLU) -> 10 |
| parametre | 101 770 — `W1` 100 352 + `b1` 128 + `W2` 1 280 + `b2` 10 |
| learning rate / batch / epoch | 0.1 / 32 / 5 |
| cross-entropy, ilk batch -> son | 2.35 -> 0.03 |
| training süresi | 0.92 sn |

Eğitmenin M-serisi Mac'inde, yalnızca CPU ile kaydedildi. Rastgelelik seed'i 0'a sabit, dolayısıyla doğruluklar yaklaşık değil kesin — yeniden koşunca aynı rakamlara oturuyor. Makineye göre değişen tek sayı süre; Colab'da farklı çıkacak. Sesli söylemeye değer iki şey: öğrenmenin neredeyse tamamı ilk epoch'ta oluyor ve 4. epoch 5.'ten yüksek alıyor — eğri iyileşmeyi bırakıp dalgalanmaya başlıyor.

`UNVERIFIED: W1[406][:4] before/now satırlarındaki sekiz rakam — hiçbir kayıtlı tabloya taşınmadı. Hücrenin göstermesi gereken şey dört sayının da kımıldadığı; rakamları bu sayfadan değil projektörden oku.`

</div>

## Daha derine

**Neden ReLU.** İki matris çarpımının arasında bir non-linearity olmazsa ağ komple çöküyor: matris çarpı matris yine bir matris, yani 784 -> 128 -> 10, tek bir 784 -> 10 katmanı kadar ifade gücüne sahip olur. ReLU bu çöküşü kıran en ucuz fonksiyon — sayı başına tek karşılaştırma, 0 veya 1 olan bir gradient. Bilinen arızası şu: girdisi hep negatif kalan bir nöronun gradient'i sonsuza kadar sıfır olur ve öğrenmeyi bırakır; transformer'larda yerini daha yumuşak varyantların alması bu yüzden. Modül 3'ün fine-tune ettiği Qwen modeli gated bir MLP'nin içinde SiLU kullanıyor; adapter'ının `gate_proj`, `up_proj` ve `down_proj`'u hedeflemesi de bu yüzden.

**101 770 nerede duruyor.** İlk katmanda 784 x 128 = 100 352 ağırlık, artı 128 bias = 100 480. Sonra 128 x 10 = 1 280, artı 10 bias = 1 290. Toplam 101 770 ve bunun %98.7'si ilk katmanda — parametreler, en geniş şeyin bir sonraki en geniş şeyle buluştuğu yerde toplanıyor.

**Transformer ölçeğinde ne değişiyor.** Kavramsal olarak neredeyse hiçbir şey. `gemma3:4b` aynı forward-loss-gradient-update döngüsü: tek dense katman yerine attention katmanları, piksel yerine metin token'ı — bizim 101 770'imize karşılık yaklaşık dört milyar parametre, kabaca 40 000 kat fazlası, pull edilmiş hâliyle diskte 3.3 GB. Canını yakan farklar ekonomik: bizim koşumuz laptop CPU'sunda bir saniyenin altı; 4B'lik bir pretraining ise kimsenin bir ücret değişti diye tekrarlamadığı bir cluster işi. Ve döngü offline; inference sırasında ağırlıklar okunuyor, asla yazılmıyor.

**10 milyon dokümanda.** Training yapmadığın için doğrudan bir maliyeti yok — ama bilgi bir kere ağırlıkların içine girdiyse, onu değiştirmek her değişiklik başına yeni bir training koşusu ve yeni bir değerlendirme demek; üç ayda bir değişen hiçbir şeyin orada işi yok.

## Çıkış cümlesi

> Training, ağırlıkları veriye fit etmek demektir. Ağırlık, training verisinin donmuş bir fotoğrafıdır.

Bu da odanın elinde bir sonraki soruyu bırakıyor: bilgi orada duruyorsa, **fotoğrafı *kendi* verimizle yeniden çekebilir miyiz?** Bu [modül 3](/tr/modules/03-finetune/) ve çalışıyor — asıl sorun da o.

<div class="presenter-note">
Bu, ağzında gevelenmemesi gereken cümle ve iki yarısının da oturması lazım. Yavaş söyle, sonra ikinci yarısını sonucuyla birlikte tekrarla: "donmuş bir fotoğraf — training bittikten sonra bilgi sayıların içinde ve yeniden eğitmedikçe o sayılar bir daha değişmiyor." Yumuşatma, continual learning'e dair bir çekince ekleme. Tahtaya yaz ve orada kalsın; modül 3, Q2 fine-tune'u ısrarla EUR 120 demeye devam ettiğinde doğrudan bu cümlenin üstüne yürüyor, modül 8 de günü onunla kapatıyor. Toplam yirmi dakika. Geride kaldıysan ve on dörde sığdırman gerekiyorsa şu sırayla kes: "Daha derine" kenar notları, loss eğrisi, tek tahmin hücresi. Tahmin sorusunu, training koşusunun kendisini, iki W1[406] satırıyla son hücreyi ve çıkış cümlesini asla kesme — zincir onlar ve modül 3 onun üstüne açılıyor.
</div>
