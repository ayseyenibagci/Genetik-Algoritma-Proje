# Genetik Algoritma ile Akıllı Tarım Projesi

Bu proje, akıllı tarım alanında görüntü işleme ve makine öğrenmesi yöntemlerinden yararlanarak buğday tarlalarındaki bitki ve yabancı otların sınıflandırılması üzerine geliştirilmiştir.

Projenin temel amacı, farklı derin öğrenme modellerinin performanslarını karşılaştırmak ve genetik algoritma kullanarak model hiperparametrelerini optimize ederek sınıflandırma performansını iyileştirmektir.

## Projenin Amacı

Tarım alanlarında yabancı otların doğru şekilde tespit edilmesi, ürün verimliliğinin artırılması ve gereksiz ilaç kullanımının azaltılması açısından önemlidir.

Bu proje kapsamında:

* Görüntü verileri üzerinde sınıflandırma çalışması yapılmıştır.
* CNN ve MobileNet tabanlı modeller kullanılmıştır.
* Farklı model yaklaşımlarının performansları karşılaştırılmıştır.
* Genetik algoritma ile hiperparametre optimizasyonu gerçekleştirilmiştir.
* Fitness değerlerinin nesiller boyunca değişimi incelenmiştir.
* En iyi kromozom ve parametre kombinasyonları belirlenmiştir.
* Model performansları confusion matrix, ROC curve ve çeşitli performans metrikleri ile değerlendirilmiştir.
* Sonuçlar grafikler kullanılarak görselleştirilmiştir.

## Kullanılan Veri Seti

Projede Kaggle üzerinde bulunan **Wheat-Weed Field Image Dataset** veri seti kullanılmıştır.

Veri seti, buğday tarlalarındaki görüntüler üzerinden buğday ve yabancı otların incelenmesine yönelik görüntüler içermektedir.

Veri setine Kaggle üzerinden ulaşabilirsiniz:

https://www.kaggle.com/datasets/sarfarazkhanmphil/wheat-weed-field-image-dataset

## Kullanılan Teknolojiler

* Python
* TensorFlow
* Keras
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* CNN
* MobileNet
* Genetik Algoritma

## Proje Yapısı

### Model Dosyaları

* `baseline_cnn.py` — Temel CNN modeli
* `cnn_augmentation.py` — Veri artırma uygulanmış CNN modeli
* `mobilenet_baseline.py` — Temel MobileNet modeli
* `mobilenet_finetune.py` — Fine-tuning uygulanmış MobileNet modeli
* `mobilenet_ga.py` — Genetik algoritma ile optimize edilen MobileNet modeli
* `ga_cnn.py` — Genetik algoritma ve CNN çalışmaları

### Genetik Algoritma

Genetik algoritma kullanılarak modelin farklı hiperparametre kombinasyonları test edilmiştir.

Optimizasyon sürecinde:

* Popülasyon oluşturma
* Fitness hesaplama
* Seçim
* Crossover
* Mutasyon
* Yeni nesil oluşturma

adımları uygulanmıştır.

Elde edilen sonuçlar nesiller boyunca takip edilerek en iyi model parametreleri belirlenmiştir.

## Görselleştirmeler

Projede genetik algoritmanın ve modellerin performansını incelemek amacıyla çeşitli grafikler oluşturulmuştur.

Örnek görselleştirmeler:

* Model karşılaştırması
* Fitness değişimi
* Popülasyon fitness dağılımı
* En iyi kromozom
* Genetik algoritma akış diyagramı
* Hiperparametre karşılaştırmaları
* Fitness histogramı
* Heatmap
* Confusion matrix
* ROC curve
* Performans metrikleri
* Crossover örneği
* Mutasyon örneği

## Sonuç

Genetik algoritma kullanılarak model hiperparametrelerinin optimize edilmesiyle farklı model yapılandırmaları karşılaştırılmış ve en uygun parametre kombinasyonunun belirlenmesi amaçlanmıştır.

Proje sonucunda elde edilen deneysel sonuçlar grafikler ve performans metrikleri kullanılarak analiz edilmiştir.

## Proje Dosyaları

Projede model eğitim kodlarının yanı sıra deney sonuçlarını içeren CSV dosyaları ve oluşturulan grafikler de bulunmaktadır.

Veri setinin kendisi proje repository'sine dahil edilmemiştir. Veri seti Kaggle üzerinden indirilebilir.

## Geliştiriciler

Bu proje akademik/proje çalışması kapsamında geliştirilmiştir.
