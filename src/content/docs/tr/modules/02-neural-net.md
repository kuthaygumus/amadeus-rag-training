---
title: "2. Sinir Ağı Nasıl Öğrenir"
description: "Bir gate değil, gate'lerin ihtiyaç duyduğu kurulum: tek bir MNIST koşusu, 101.770 sayı ve günün geri kalanının üstüne oturduğu cümle."
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate değil — bir kurulum

> **Bir model 'öğrenmek' derken ne yapıyor?**

Bugünkü diğer bütün modüller bir aracın patlamasıyla açılıyor ve bir sonraki aracı zorunlu kılan hatayla kapanıyor. Bu modül öyle değil ve bunu süslemek yerine açıkça söylemek gerek. Burada bir şey patlamıyor, Helios corpus'una karşı bir ölçüm yapılmıyor, yeni bir retrieval tekniği tanıtılmıyor. Modül 2 tek bir cümleyi hak etmek için var — *ağırlık donmuş bir fotoğraftır* — ve modül 3, 4 ve 9 bu cümleye yaslanıyor, hiçbiri de durup onu türetecek zamana sahip değil.

## Hâlâ içinde durduğumuz hata

On dakika önce, aynı laptop, aynı model, iki soruda da context yok, doküman yok:

**`Helios CLASSIC K iptal cezası?`** → model cezanın genelde ücretin **"20-30% ceza"**'sı olduğunu anlattı.
**`K booking class typical penalty?`** → **"K (Business) sınıfı %10-20"**, iki kere yanlış: aralık uydurma ve K business değil, indirimli bir economy booking class.
**`What time does H9 1487 depart?`** → temiz bir şekilde bilmediğini söyledi.

Gerçek rakam `corpus/2026-Q3/fare_classic_shorthaul.md` dosyasında, K satırında: **EUR 90**, sabit, yolcu ve yön başına. Geçen çeyrek aynı satır EUR 120 diyordu.

Odanın refleksi "model uydurdu" demek. Bu bir tarif, açıklama değil ve bir sonraki adıma karar vermeye yetmiyor. Bir saat sonra iki seçenek arasında karar vereceğiz: bu modeli kural kitabıyla fine-tune etmek mi, yoksa kural kitabını sorgu anında retrieve etmek mi. 20-30 sayısının fiziksel olarak nerede durduğunu söyleyemiyorsan, EUR 90'ın neden aynı yerden gelemeyeceğini de söyleyemezsin; o zaman bu karar dürüst bir karar olmaz.

O yüzden kutuyu açıyoruz. Deep learning öğretmek için değil — bu bir günlük RAG eğitimi — saat 15:00'e kadar altı kere ihtiyacın olacak tek bir cümleyi hak etmek için.

<div class="presenter-note">
Bu modülün ajandadaki yeri 09:32–09:52. Dosyayı açmadan önce odaya sor: "model kurgusal bir havayolu için 20-30% uydurdu. Böyle bir sayı modelin içinde fiziksel olarak nerede duruyor?" İki üç cevap al. Biri "training verisinde" diyecek — itiraz et, training verisi yok, eğitim bitince atıldı. Biri "ağırlıklarda" diyecek — o zaman sor, ağırlık nedir? Oda genelde burada susar. Bu modül tam olarak o sessizlik için var. 90 saniye, fazlası değil.
</div>

## Makine gerçekte ne yapıyor

Hâlâ dürüst kalan en küçük ağı eğitiyoruz: girdi el yazısı rakam, çıktı bir rakam. 784 girdi, ReLU'lu 128 nöronluk tek bir gizli katman, 10 çıktı. **101.770 parametre.**

Bir MNIST görüntüsü 28x28 gri piksel. Düzleştirince elinde 0 ile 1 arasında 784 sayıdan oluşan bir vektör kalıyor. Ağın gördüğü tek şey bu vektör; görüntü diye bir şeyden haberi yok.

**Forward pass.** 784'lük vektörü 784x128'lik ağırlık matrisiyle çarp, 128 bias ekle, 128 sayı çıkar. Negatif olanların hepsini sıfırla — ReLU bu, fonksiyonun tamamı `max(0, x)`. O 128 sayıyı 128x10'luk matrisle çarp, 10 bias ekle, 10 sayı çıkar. Bunları olasılığa çevir; en büyüğü cevap. Modelin tamamı bu. İki matris çarpımı ve bir kırpma.

**Loss.** O cevabın ne kadar yanlış olduğunu söyleyen tek bir sayı: model yanılıyorsa büyük, tutturuyorsa küçük. Eğitimden önce ağ olasılığı on rakama aşağı yukarı eşit dağıtıyor ve koşumuzun ilk batch'i **2.35** loss alıyor — tahmin etmenin maliyetine yakın bir değer. Kaydettiğimiz son batch ise **0.03** alıyor.

**Gradient.** Buradaki tek gerçek fikir. 101.770 parametrenin her biri için, o parametre birazcık artsa loss'un ne kadar değişeceğini hesaplayabiliyoruz. Bu 101.770 eğimin tamamı gradient; backpropagation da zincir kuralının, hepsini yaklaşık bir forward pass maliyetine çıkaracak kadar verimli uygulanmış hali.

**Update.** Her parametreyi loss'u düşüren yönde küçük bir adım kaydır, sonraki mini-batch'i al, tekrar yap. Altmış bin görüntü, beş tur. Başka hiçbir şey olmuyor: akıl yürütme adımı yok, saklanan örnek yok, lookup yok. Training, loss düşmeyi bırakana kadar 101.770 sayıyı dürten bir döngü.

<div class="presenter-note">
Eğitim bloğunu çalıştırmadan önce tahmin iste: "bu laptopta, GPU yok, PyTorch yok, 60.000 görüntü üzerinde 5 epoch — ne kadar sürer?" Üç kişi sesli olarak tahminini söylesin. Genelde dakika derler. Sonra çalıştır: bu sayıların ölçüldüğü makinede **0.85 saniye**. Tahminle kronometre arasındaki fark sonraki yirmi dakikayı oturtan şey. Cümleni bitirmene kalmadan bitiyor, o yüzden üstüne konuşmaya çalışma — çalıştır, sessizliğin oturmasına izin ver, sonraki blokta loss eğrisini bastır ve asıl onu anlat. Buradaki tek donanıma bağlı sayı süre; odadaki bir laptop birkaç saniye sürerse bunu söyle ve devam et. Bu dosya Ollama istemiyor ve hiçbir network çağrısı yapmıyor, dolayısıyla tek gerçek arıza ihtimali MNIST cache'inin yerinde olmaması — o durumda da preflight traceback yerine seed komutunu basıyor. Sahnede olursa aşağıdaki ölçüm tablosunu bu sayfadan oku ve devam et, odanın önünde debug etme.
</div>

## Neden bu konfigürasyon, daha küçüğü değil

Sınıfta akla gelen kestirme 6.000 görüntü ve 3 epoch. **0.05 sn**'de bitiyor ve **%91.67**'ye çıkıyor. Tam koşu aynı makinede bir saniyenin altında sürdüğü için bu kestirme hiçbir şey kazandırmıyor, karşılığında **5.8 puan** veriyor — ve rakamlarda %91.67, arka sıradaki şüphecinin "bu gerçekten çalışmıyor" demesini haklı çıkaracak kadar zayıf.

Tam koşu **0.85 sn** sürüyor ve doğruluğu eğitimden önceki **%9.87**'den bir epoch sonra **%95.35**'e, beş epoch sonra **%97.47**'ye taşıyor. Asıl mesele bu eğri: öğrenmenin neredeyse tamamı ilk geçişte oluyor, kalan dört epoch iki puan için boğuşuyor. Dördüncü epoch **%97.62** alıyor, yani beşinciden yüksek — eğri iyileşmeyi bırakıp dalgalanmaya başlıyor, ve bunu saklamak yerine sesli söylemek gerek.

Her şey numpy üzerinde koşuyor. Burada PyTorch yok ve günün geri kalanının zaten istediğinin ötesinde kurulacak bir şey yok; bağımlılıklar tam olarak `numpy` ve `chromadb`. Ayrıca backward pass'in her satırı bir framework çağrısının arkasında değil, dosyada görünür oluyor.

## Ne çalıştırıyorsun

`notebooks/01_mnist_tiny_net.py` dosyasını VS Code'da aç ve blokları Shift+Enter ile çalıştır (Microsoft Python eklentisi). Bu eğitimdeki notebook'lar editörün içinde çalışan percent formatında `.py` dosyaları; kurulacak bir notebook sunucusu yok.

Dataset bir kere, evde, indirmeye izin veren bir ağda yerleştiriliyor:

```bash
python scripts/seed_offline_assets.py
```

Bu komut `notebooks/mnist_data/` altına dört `.gz` arşivi yazıyor, toplam **11.6 MB** (11.594.722 byte). Notebook'un kendisi hiçbir network çağrısı yapmıyor; ilk blok bu dört dosyayı diskten okuyor ve dosyalar yoksa preflight traceback yerine yukarıdaki komutu basıp duruyor.

**Ne görmen gerekiyor**, blok blok:

1. MNIST yükleme — `train (60000, 784)   test (10000, 784)   (read from mnist_data/, no network)`
2. ASCII olarak basılmış bir training görüntüsü, `label: 3`
3. dört dizi — `W1 (784, 128)`, `b1`, `W2 (128, 10)`, `b2` — ve `total: 101,770 numbers`
4. forward pass, ardından `accuracy before any training: 9.9%`
5. eğitim döngüsü: beş epoch satırı, sonuncusu `epoch 5: test accuracy 97.47%`, ardından `trained in 0.85 seconds on a laptop CPU`
6. ASCII loss eğrisi — `first batch: loss 2.35     last: loss 0.03`
7. tek bir test görüntüsü, `label: 7  predicted: 7  confidence: 99.9%`
8. özet — mimari ve parametre sayısı `(unchanged)`, `W1[0][:4]` basılıyor, `accuracy: 97.47%   (was 9.9%)`

**Ne kadar sürer.** Dosyanın tamamı baştan sona birkaç saniyede koşuyor; eğitim bloğu bunun 0.85 sn'si. Seed adımı ise bir gün önce akşam yapılan tek seferlik bir indirme.

**Asıl önemli blok 8.** Mimari değişmedi. Parametre sayısı değişmedi. Ortaya bir veritabanı çıkmadı, hiçbir görüntü saklanmadı, o dört dizinin dışına hiçbir şey yazılmadı — ama doğruluk %9.87'den %97.47'ye çıktı. Ağın altmış bin el yazısı rakam hakkında öğrendiği her şey, `W1[0][:4]`'ün eskiden tuttuğu sayılarla şimdi tuttuğu sayılar arasındaki farktan ibaret. Bu, training setinin döngü bittiğinde bir kere çekilmiş fotoğrafı ve yarın da aynı görünecek.

## Sayılar ne dedi

<div class="measured">

| konfigürasyon | görüntü | epoch | süre | doğruluk |
|---|---|---|---|---|
| sınıf kestirmesi | 6.000 | 3 | 0.05 sn | %91.67 |
| **çalıştırdığımız** | **60.000** | **5** | **0.85 sn** | **%9.87 -> %97.47** |

| epoch | test doğruluğu |
|---|---|
| eğitimden önce | %9.87 |
| 1 | %95.35 |
| 2 | %96.45 |
| 3 | %97.26 |
| 4 | %97.62 |
| 5 | %97.47 |

| mimari | değer |
|---|---|
| şekil | 784 -> 128 (ReLU) -> 10 |
| parametre | 101.770 |
| ilk batch loss -> son | 2.35 -> 0.03 |

Tekrar üretmek için: `python notebooks/01_mnist_tiny_net.py`, ya da blokları VS Code'da çalıştır. rng seed'i 0'a sabit, dolayısıyla doğruluklar buradaki değerlere yuvarlama basamağı farkıyla oturuyor. Süre bir M-serisi laptop CPU'sunda ölçüldü ve makineye göre değişen tek sayı o.

</div>

## Daha derine

**Neden ReLU.** İki matris çarpımının arasında bir non-linearity olmazsa ağ komple çöküyor: matris çarpı matris yine bir matris, yani 784 -> 128 -> 10, tek bir 784 -> 10 katmanı kadar ifade gücüne sahip olur ve 128 gizli nöron hiçbir şey kazandırmaz. ReLU bu çöküşü kıran en ucuz fonksiyon: sayı başına tek karşılaştırma, gradient'i 0 veya 1 olduğu için geri yolda hiçbir şey küçülmüyor ve büyük pozitif girdilerde doyuma gitmiyor — derin yığınları eğitilebilir yapan da bu oldu. Bilinen arızası şu: girdisi hep negatif kalan bir nöronun gradient'i sonsuza kadar sıfır olur ve öğrenmeyi bırakır. GELU ve leaky ReLU bu yüzden var, transformer'ların bunun yerine GELU kullanmasının sebebi de bu.

**101.770 nerede duruyor.** İlk katmanda 784 x 128 = 100.352 ağırlık, artı 128 bias = 100.480. Sonra 128 x 10 = 1.280, artı 10 bias = 1.290. Toplam 101.770 ve bunun %98.7'si ilk matriste — parametreler, en geniş şeyin bir sonraki en geniş şeyle buluştuğu yerde toplanıyor. Bu aynı zamanda bir katmanın fiyatını da veriyor: gizli katmanı 256'ya çıkarmak 100.480 parametre daha demek, ikinci bir 128'lik gizli katman eklemek ise 16.512. Genişlik pahalı, derinlik ucuz — çoğu insanın geldiği sezgi bu değil.

**Doğruluk neden plato yapıyor.** %95.35 ile %97.47 arasında kalan hatalar sistematik olmaktan çıkıp gerçekten belirsiz rakamlara dönüşüyor — bire benzeyen yediler, dokuza benzeyen dörtler. Gradient descent ortalama loss'u düşürüyor, yani bütçesini kütlenin olduğu yere harcıyor; son iki puan kuyrukta ve oradan birini düzelten her dürtüş başka bir şeyi biraz bozuyor. Dördüncü epoch'un beşinciden yüksek çıkmasının sebebi de bu. Daha fazla epoch seni bunun ötesine geçirmiyor. Modelin ifade edebildiği şeyi değiştirmek geçiriyor — örneğin convolution, çünkü bir fırça darbesinin karenin neresinde olursa olsun aynı şeyi ifade ettiğini mimariye gömüyor. Aynı mimarinin daha genişi ise çoğunlukla training setini daha hızlı ezberliyor.

**Transformer ölçeğinde ne değişiyor.** Kavramsal olarak neredeyse hiçbir şey. `qwen2.5:3b`, aynı forward-loss-gradient-update döngüsü: tek dense katman yerine attention katmanları, piksel yerine metin token'ı — bizim 101.770'imize karşılık üç milyar parametre, kabaca 30.000 kat fazlası ve kurulu hâlde diskte 1.9 GB. Canını yakan farklar ekonomik: bizim koşumuz laptop CPU'sunda 0.85 sn, 3B'lik bir pretraining koşusu ise kimsenin bir ücret değişti diye tekrarlamadığı bir cluster işi. Ve döngü offline; inference sırasında ağırlıklar okunuyor, asla yazılmıyor.

**10 milyon dokümanda.** Training yapmadığın için doğrudan bir maliyeti yok. Ama bu resim, seçeneği eleyen şeyin ta kendisi: bilgi bir kere ağırlıkların içine girdiyse, onu değiştirmek her değişiklik başına yeni bir training koşusu ve yeni bir değerlendirme demek. Üç ayda bir değişen hiçbir şeyin orada işi yok. Bunu bir modül boyunca aklında tut.

## Çıkış cümlesi

> Training = ağırlıkları veriye fit etmek. Ağırlık artık donmuş bir fotoğraf.

Bu da odanın elinde bir sonraki soruyu bırakıyor: bilgi orada duruyorsa, **kendi verimi o ağırlıkların içine nasıl sokarım?** Modül 3 bu, ve çalışıyor — asıl sorun da o.

<div class="presenter-note">
Bu, ağzında gevelenmemesi gereken cümle ve iki yarısının da oturması lazım. Yavaş söyle, sonra ikinci yarısını sonucuyla birlikte tekrarla: "donmuş bir fotoğraf — training bittikten sonra bilgi sayıların içinde ve yeniden eğitmedikçe o sayılar bir daha değişmiyor." Yumuşatma, continual learning kaydı düşme. Tahtaya yaz ve orada kalsın; modül 3, Q2 fine-tune'u ısrarla EUR 120 demeye devam ettiğinde doğrudan bu cümlenin üstüne yürüyor. Modülün tamamı yirmi dakika, 09:32–09:52. Geride kaldıysan ve bunu on dörde sığdırman gerekiyorsa şu sırayla kes: "Daha derine" kenar notları, blok 6 (loss eğrisi) ve blok 7 (tek tahmin), sonra 6.000 görüntülük karşılaştırma. Tahmin sorusunu, eğitim koşusunun kendisini, blok 8'i ve çıkış cümlesini asla kesme — zincir onlar ve modül 3 onun üstüne açılıyor.
</div>
