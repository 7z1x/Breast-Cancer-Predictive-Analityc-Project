# Laporan Proyek Machine Learning - Zulfahmi M. Ardianto

## Domain Proyek

### Latar Belakang

Kanker payudara adalah salah satu jenis kanker paling umum di dunia, dengan lebih dari 2 juta kasus baru setiap tahunnya [1]. Deteksi dini melalui analisis fitur tumor (seperti radius, tekstur, dan kekompakan) sangat penting untuk meningkatkan tingkat kelangsungan hidup pasien. Dataset Breast Cancer Wisconsin menyediakan data pengukuran tumor yang dapat digunakan untuk mengklasifikasikan tumor sebagai malignant (kanker) atau benign (non-kanker) menggunakan pendekatan machine learning. Masalah ini relevan karena diagnosis yang akurat dapat membantu dokter membuat keputusan pengobatan yang tepat, mengurangi risiko kesalahan diagnosis, dan meningkatkan efisiensi proses medis.


### Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan
Masalah klasifikasi tumor payudara harus diselesaikan untuk:

- Meningkatkan Akurasi Diagnosis: Mengurangi kesalahan manusia dalam menentukan jenis tumor.

- Mendukung Keputusan Medis: Memberikan alat bantu bagi dokter untuk diagnosis cepat dan akurat.

- Efisiensi Waktu: Mengotomatisasi proses analisis fitur tumor, yang biasanya memakan waktu jika dilakukan secara manual.

Pendekatan machine learning dipilih karena kemampuannya untuk mengenali pola kompleks dalam data numerik, seperti yang ada pada dataset ini. Dengan melatih model seperti Logistic Regression, Random Forest, dan SVM, kita dapat membangun sistem prediksi yang akurat dan terukur.

## Business Understanding
### Problem Statement:
- Bagaimana cara mengklasifikasikan tumor payudara sebagai malignant atau benign berdasarkan fitur pengukuran tumor dengan akurasi tinggi?
- Algoritma machine learning mana yang memberikan performa terbaik untuk tugas klasifikasi ini?
- Bagaimana memastikan model yang dibangun dapat dipercaya untuk mendukung keputusan medis

### Goals:
- Membangun model machine learning dengan akurasi minimal 95% untuk mengklasifikasikan tumor payudara.
- Membandingkan performa tiga algoritma (Logistic Regression, Random Forest, SVM) dengan dua metode seleksi fitur (SelectKBest dan RFE) untuk memilih model terbaik.
- Menyediakan evaluasi yang terukur menggunakan metrik seperti F1 Score, ROC-AUC, dan PR-AUC untuk memastikan keandalan model.
### Solution Statements:
Untuk mencapai tujuan, dua solusi diusulkan:
1. Menggunakan Berbagai Algoritma dengan Seleksi Fitur:
- Melatih tiga algoritma (Logistic Regression, Random Forest, SVM) dengan fitur yang dipilih menggunakan SelectKBest dan RFE.
- Metrik evaluasi: Akurasi, Precision, Recall, F1 Score, ROC-AUC, dan PR-AUC.
- Alasan: Membandingkan algoritma memungkinkan pemilihan model terbaik berdasarkan performa.

## Data Understanding
### **Informasi Data**

- Sumber: Breast cancer prediction Dataset dari [Kaggle](https://www.kaggle.com/code/buddhiniw/breast-cancer-prediction) dan bersumber dari akun kaggle yaitu buddhiniw.
- Jumlah Data: 569 baris (pengamatan) dan 33 kolom (32 fitur + 1 kolom kosong).
- Kondisi Data:
  - Tidak ada missing value pada fitur utama.
  - Kolom Unnamed: 32 sepenuhnya kosong, kolom id tidak relevan untuk modeling.
- Target: Kolom diagnosis dengan nilai M (malignant) dan B (benign).

![](https://github.com/7z1x/Breast-Cancer-Predictive-Analityc-Project/blob/e3e91b45a651c8cf8080cd9c8a9541b74e4db7e4/image/Data%20Loading.jpg)

### **Variabel/Fitur:**
- Total: 30 fitur numerik, 1 fitur target kategorik (diagnosis), 1 kolom ID, dan 1 kolom kosong.
  
| Kolom                    | Deskripsi                                                                 | Non-null Count | dtype   |
|--------------------------|---------------------------------------------------------------------------|----------------|---------|
| id                       | ID unik untuk tiap pasien                                                 | 569            | int64   |
| diagnosis                | Hasil diagnosis: M (ganas) atau B (jinak)                                 | 569            | object  |
| radius_mean              | Rata-rata jari-jari sel inti                                              | 569            | float64 |
| texture_mean             | Rata-rata variasi tingkat abu-abu (tekstur permukaan sel)                 | 569            | float64 |
| perimeter_mean           | Rata-rata keliling sel inti                                               | 569            | float64 |
| area_mean                | Rata-rata luas permukaan sel inti                                         | 569            | float64 |
| smoothness_mean          | Rata-rata kehalusan batas sel                                             | 569            | float64 |
| compactness_mean         | Rata-rata kekompakan sel                                                  | 569            | float64 |
| concavity_mean           | Rata-rata tingkat cekungan batas sel                                      | 569            | float64 |
| concave points_mean      | Rata-rata jumlah titik-titik cekungan di batas sel                        | 569            | float64 |
| symmetry_mean            | Rata-rata simetri bentuk sel                                              | 569            | float64 |
| fractal_dimension_mean   | Rata-rata kompleksitas batas sel                                          | 569            | float64 |
| radius_se                | Standar deviasi jari-jari sel                                             | 569            | float64 |
| texture_se               | Standar deviasi tekstur sel                                               | 569            | float64 |
| perimeter_se             | Standar deviasi keliling sel                                              | 569            | float64 |
| area_se                  | Standar deviasi luas sel                                                  | 569            | float64 |
| smoothness_se            | Standar deviasi kehalusan                                                 | 569            | float64 |
| compactness_se           | Standar deviasi kekompakan                                                | 569            | float64 |
| concavity_se             | Standar deviasi tingkat cekungan                                          | 569            | float64 |
| concave points_se        | Standar deviasi jumlah titik cekungan                                     | 569            | float64 |
| symmetry_se              | Standar deviasi simetri sel                                               | 569            | float64 |
| fractal_dimension_se     | Standar deviasi kompleksitas batas sel                                    | 569            | float64 |
| radius_worst             | Nilai maksimum radius dari 3 sel terburuk                                 | 569            | float64 |
| texture_worst            | Nilai maksimum tekstur                                                    | 569            | float64 |
| perimeter_worst          | Nilai maksimum keliling                                                   | 569            | float64 |
| area_worst               | Nilai maksimum luas                                                       | 569            | float64 |
| smoothness_worst         | Nilai maksimum kehalusan                                                  | 569            | float64 |
| compactness_worst        | Nilai maksimum kekompakan                                                 | 569            | float64 |
| concavity_worst          | Nilai maksimum tingkat cekungan                                           | 569            | float64 |
| concave points_worst     | Nilai maksimum jumlah titik cekungan                                      | 569            | float64 |
| symmetry_worst           | Nilai maksimum simetri                                                    | 569            | float64 |
| fractal_dimension_worst  | Nilai maksimum kompleksitas batas sel                                     | 569            | float64 |
| Unnamed: 32              | Kolom kosong                                                              | 0              | float64 |

### **Distribusi data label:**
| Diagnosis | Jumlah |
|-----------|--------|
| 0 (Benign)   | 357    |
| 1 (Malignant) | 212    |

- Distribusi target menunjukkan lebih banyak kasus benign (63%) dibandingkan malignant (37%), tetapi tidak terlalu imbalanced

![](https://github.com/7z1x/Breast-Cancer-Predictive-Analityc-Project/blob/e3e91b45a651c8cf8080cd9c8a9541b74e4db7e4/image/Dsitribusi%20fitur%20taget.jpg)

###  Korelasi antar fitur (heatmap)
![](https://github.com/7z1x/Breast-Cancer-Predictive-Analityc-Project/blob/e3e91b45a651c8cf8080cd9c8a9541b74e4db7e4/image/heatmap_korelasi.png)

## Data Preparation

### Menangani Missing Value dan Fitur Nonrelevan
- Melakukan tindakan drop untuk kolom yang tidak relevan seperti Id dan data yang kosong (Unnamed: 32)
### Prepare data target
- Data target memiliki 2 nilai yaitu 'M': malignant, 'B': benign
- Mengubah data target atau encoding menjadi numerik yaitu 1 untuk malignant(M) atau kanker ganas dan 0 Benign(B) atau kanker jinak

### Pembagian Data
- Memisahkan fitur dan Target
<pre> <code> X = df.drop('diagnosis', axis=1) 
  y = df['diagnosis'] </code> </pre>
- Split data (80:20) dengan stratifikasi untuk menjaga proporsi kelas.
<pre> <code> X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
</code> </pre>
- Standarisasi Fitur:
<pre> <code> 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 
</code> </pre>

## Feature Selection
-  Untuk mengurangi dimensi, multikolinearitas dan meningkatkan efisiensi model, saya melakukan feature selection menggunakan SelectKBest untuk memilih 10 fitur terbaik berdasarkan skor ANOVA dan RFE untuk memilih 10 fitur menggunakan eliminasi rekursif dengan tiga estimator.
-  SelectKBest:
SelectKBest adalah metode seleksi fitur univariat yang memilih fitur-fitur terbaik berdasarkan skor statistik seperti chi-squared atau f-classif. Metode ini menunjukkan bagaimana reduksi dimensi atau seleksi fitur meningkatkan kinerja atau akurasi algoritma klasifikasi[2].
- Recursive Feature Elimination (RFE) adalah teknik seleksi fitur berbasis pembungkus yang menggunakan model pembelajaran mesin untuk menentukan skor relevansi fitur[3].
### SelectKBest
<pre> <code>
k = 10  # Jumlah fitur yang diambil
skb = SelectKBest(score_func=f_classif, k=k)
X_train_skb = skb.fit_transform(X_train_scaled, y_train)
X_test_skb = skb.transform(X_test_scaled)
skb_features = X.columns[skb.get_support()]
print("Fitur dari SelectKBest:", skb_features.tolist())
</code> </pre>
- Fitur dari SelectKBest: ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 'radius_worst', 'perimeter_worst', 'area_worst', 'concavity_worst', 'concave points_worst']
  
<br>


### Recursive Feature Elimination (RFE)
1. Logistic Regression
<pre> <code>
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=k)
rfe.fit(X_train_scaled, y_train)
X_train_rfe = rfe.transform(X_train_scaled)
X_test_rfe = rfe.transform(X_test_scaled)
rfe_features = X.columns[rfe.support_]
print("Fitur dari RFE LR:", rfe_features.tolist())
</code> </pre>
**Insight:**
- Fitur LR terpilih mencakup 'concave points_mean', 'radius_se', 'area_se', 'compactness_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'concavity_worst', 'concave points_worst', yang konsisten dengan korelasi tinggi terhadap target.
  
<br>


2. Random Forest
<pre> <code>
k = 10
rfe_rf = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=k)
rfe_rf.fit(X_train_scaled, y_train)
X_train_rfe_rf = rfe_rf.transform(X_train_scaled)
X_test_rfe_rf = rfe_rf.transform(X_test_scaled)
rfe_features_rf = X.columns[rfe_rf.support_]
print("Fitur dari RFE (Random Forest):", rfe_features_rf.tolist()
</code> </pre>

**Insight:**
- Fitur RF terpilih mencakup 'radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'concave points_worst', yang konsisten dengan korelasi tinggi terhadap target.
  
<br>

3. Suport Vector Machine(SVM)
<pre> <code>
rfe_svm = RFE(estimator=SVC(kernel="linear", random_state=42), n_features_to_select=k)
rfe_svm.fit(X_train_scaled, y_train)
X_train_rfe_svm = rfe_svm.transform(X_train_scaled)
X_test_rfe_svm = rfe_svm.transform(X_test_scaled)
rfe_features_svm = X.columns[rfe_svm.support_]
print("Fitur dari RFE (SVM):", rfe_features_svm.tolist())
</code> </pre>

**Insight:**
- Fitur SVM terpilih mencakup 'concavity_mean', 'concave points_mean', 'radius_se', 'texture_se', 'area_se', 'compactness_se', 'fractal_dimension_se', 'texture_worst', 'area_worst', 'concavity_worst', yang konsisten dengan korelasi tinggi terhadap target.
 
<br>

### Modeling
Dalam proyek ini, digunakan tiga algoritma machine learning untuk melakukan klasifikasi diagnosis kanker payudara, yaitu Logistic Regression, Random Forest Classifier, dan Support Vector Machine (SVM). Selain itu, diterapkan dua metode seleksi fitur: SelectKBest dan Recursive Feature Elimination (RFE), untuk meningkatkan akurasi dan efisiensi model. Pemilihan algoritma dan metode seleksi fitur didasarkan pada kemampuannya menangani data numerik dan klasifikasi biner secara efektif.

- ***Logistic Regression:***<br>
Logistic Regression adalah model klasifikasi linier yang digunakan untuk memprediksi probabilitas dari kelas target biner. Optimasi adaptif dari hyperparameter mengurangi biaya komputasi dalam memilih nilai hyperparameter yang baik, dan memungkinkan nilai optimal ini ditentukan lebih tepat[4]. Model ini sederhana namun kuat untuk baseline, dan mampu memberikan interpretasi koefisien yang jelas untuk setiap fitur. Dalam proyek ini, Logistic Regression dioptimalkan menggunakan regularisasi L2 sebagai default dari Scikit-learn.

- ***Random Forest Classifier:***<br>
Random forests adalah kombinasi dari prediktor pohon di mana setiap pohon bergantung pada nilai dari vektor acak yang diambil secara independen dan dengan distribusi yang sama untuk semua pohon dalam hutan[5]. Model ini tahan terhadap overfitting dan mampu menangkap pola yang kompleks dalam data dan random_state=42 untuk mengatur seednya. 

- ***Support Vector Machine (SVM):***<br>
Support vector machine adalah sistem untuk melatih mesin pembelajaran linier secara efisien dalam ruang fitur yang diinduksi oleh kernel[6]. Dalam proyek ini, digunakan kernel="linear"  yang cocok untuk dataset berdimensi tinggi dan sparsitas rendah serta menggunakan random_state=42 untuk mengatur seednya.

***Metrik***<br>
| Metrik      | Rumus                                      | Penjelasan                                                                                  |
|-------------|--------------------------------------------|---------------------------------------------------------------------------------------------|
| Akurasi     | (TP + TN) / (TP + TN + FP + FN)            | Mengukur proporsi prediksi benar; memberikan gambaran umum performa model.                  |
| Presisi     | TP / (TP + FP)                             | Mengukur ketepatan prediksi positif.                                                        |
| Recall      | TP / (TP + FN)                             | Mengukur kemampuan mendeteksi positif sebenarnya; penting untuk kelas minoritas.            |
| F1-score    | 2 * (Presisi * Recall) / (Presisi + Recall)| Menyeimbangkan presisi dan recall; cocok untuk dataset yang tidak seimbang.                 |
 
<br>

### SelectKBest
| Model                        | Accuracy | Precision | Recall | F1 Score |
|-----------------------------|----------|-----------|--------|----------|
| SelectKBest + Logistic Regression | 0.9561   | 0.9744    | 0.9048 | 0.9383   |
| SelectKBest + Random Forest       | 0.9561   | 1.0000    | 0.8810 | 0.9367   |
| SelectKBest + SVM                 | 0.9737   | 1.0000    | 0.9286 | 0.9630   |

### RFE
| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| RFE + Logistic Regression | 0.9737   | 0.9756    | 0.9524 | 0.9639   |
| RFE + Random Forest       | 0.9737   | 1.0000    | 0.9286 | 0.9630   |
| RFE + SVM                 | 0.9474   | 0.9737    | 0.8810 | 0.9250   |

<br>

## Evaluasi
- Confusion Matrix menawarkan teknik yang mendalam dan terperinci untuk mengevaluasi kinerja klasifikator, yang penting dalam ilmu data[7]
- Confusion Matrix: Menampilkan TP, TN, FP, FN.
<br>

| Confusion Matrix      | Penjelasan                                                                 |
|-----------------------|----------------------------------------------------------------------------|
| *True Positive (TP)*  | Jumlah prediksi positif yang benar terhadap jumlah positif yang sebenarnya |
| *False Positive (FP)* | Jumlah prediksi positif yang salah                                         |
| *True Negative (TN)*  | Jumlah prediksi negatif yang benar terhadap jumlah negatif yang sebenarnya |
| *False Negative (FN)* | Jumlah prediksi negatif yang salah                                         |

| Model                       | Accuracy | Precision | Recall | F1 Score |
|-----------------------------|----------|-----------|--------|----------|
| RFE + Logistic Regression    | 0.9737   | 0.9756    | 0.9524 | 0.9639   |
| SelectKBest + SVM            | 0.9737   | 1.0000    | 0.9286 | 0.9630   |
| RFE + Random Forest          | 0.9737   | 1.0000    | 0.9286 | 0.9630   |
| SelectKBest + Logistic Regression | 0.9561   | 0.9744    | 0.9048 | 0.9383   |
| SelectKBest + Random Forest  | 0.9561   | 1.0000    | 0.8810 | 0.9367   |
| RFE + SVM                    | 0.9474   | 0.9737    | 0.8810 | 0.9250   |

<br>

| Model                          | True Neg (TN) | False Pos (FP) | False Neg (FN) | True Pos (TP) |
|--------------------------------|---------------|----------------|----------------|---------------|
| SelectKBest + Logistic Regression | 71            | 1              | 4              | 38            |
| SelectKBest + Random Forest       | 72            | 0              | 5              | 37            |
| SelectKBest + SVM                 | 72            | 0              | 3              | 39            |
| RFE + Logistic Regression         | 71            | 1              | 2              | 40            |
| RFE + Random Forest               | 72            | 0              | 3              | 39            |
| RFE + SVM                         | 71            | 1              | 5              | 37            |


<br>

![](https://github.com/7z1x/Breast-Cancer-Predictive-Analityc-Project/blob/e3e91b45a651c8cf8080cd9c8a9541b74e4db7e4/image/Evaluasi%20performa%20model.jpg)

<br>

**Insight Evaluasi Model:**

1. Berdasarkan evaluasi enam kombinasi model dan metode seleksi fitur, model RFE + Logistic Regression memberikan performa paling seimbang dengan F1 Score tertinggi (0.9639), diikuti sangat dekat oleh SelectKBest + SVM dan RFE + Random Forest yang juga memiliki akurasi dan precision 100%, namun sedikit lebih rendah di aspek recall.

2. Metode seleksi fitur RFE (Recursive Feature Elimination) terbukti lebih konsisten memberikan performa tinggi dibandingkan SelectKBest. Di sisi lain, algoritma SVM dan Random Forest mampu mencapai precision sempurna, namun trade-off-nya adalah recall sedikit lebih rendah (mungkin cenderung overfitting pada data benign).

3. Secara keseluruhan, model RFE + Logistic Regression dapat direkomendasikan karena memberikan keseimbangan terbaik antara semua metrik evaluasi.

<br>

### Menggunakan Metrik evaluasi ROC-AUC dan PR-AUC
| Model                        | ROC-AUC | PR-AUC  |
|------------------------------|---------|---------|
| RFE + Logistic Regression     | 0.9974  | 0.9957  |
| RFE + Random Forest           | 0.9945  | 0.9916  |
| RFE + SVM                     | 0.9944  | 0.9913  |
| SelectKBest + Logistic Regression | 0.9974  | 0.9956  |
| SelectKBest + Random Forest   | 0.9894  | 0.9848  |
| SelectKBest + SVM             | 0.9987  | 0.9978  |
<br>

**Insight Metrik Evaluasi ROC-AUC dan PR-AUC :**

1. Model terbaik secara ROC-AUC dan PR-AUC adalah SelectKBest + SVM

   - ROC-AUC: 0.9987

   - PR-AUC : 0.9978
  
2. Model ini menunjukkan kemampuan terbaik dalam membedakan antara kelas dan tetap sangat baik dalam situasi data tidak seimbang.

3. Model lain yang juga sangat kuat:

   - RFE + LR (ROC-AUC: 0.9974, PR-AUC: 0.9957)

   - SelectKBest + Logistic Regression (ROC-AUC: 0.9974, PR-AUC: 0.9956)
Kedua model ini sangat kompetitif, hanya sedikit di bawah SVM.

   - Model dengan performa paling rendah dalam metrik ini:

   - SelectKBest + Random Forest (ROC-AUC: 0.9894, PR-AUC: 0.9848)
Meski tetap tinggi, performanya sedikit di bawah yang lain.

## Kesimpulan
Proyek ini berhasil membangun model klasifikasi untuk mendiagnosis kanker payudara dengan akurasi tinggi menggunakan dataset Breast Cancer Wisconsin. Berdasarkan analisis confusion matrix dan metrik evaluasi, **RFE + Logistic Regression** adalah model terbaik, diikuti oleh **SelectKBest + SVM** yang juga menunjukkan performa luar biasa.

**Penjelasan Confusion Matrix**: Confusion matrix menggambarkan performa model melalui True Negative (TN, *benign* diprediksi *benign*), False Positive (FP, *benign* diprediksi *malignant*), False Negative (FN, *malignant* diprediksi *benign*), dan True Positive (TP, *malignant* diprediksi *malignant*). Dalam konteks medis, meminimalkan FN sangat kritis untuk memastikan kasus *malignant* tidak terlewat, sementara FP yang rendah menjaga kepercayaan pada prediksi *malignant*.

**Alasan Pemilihan RFE + Logistic Regression**:
- **False Negatives Terendah**: Hanya 2 FN (dari 42 kasus *malignant*), menghasilkan recall tertinggi (0.9524), memastikan deteksi 40 kasus *malignant*.
- **Precision Tinggi**: 0.9756 (1 FP), menunjukkan prediksi *malignant* yang andal.
- **F1 Score Kompetitif**: 0.9639, menunjukkan keseimbangan precision dan recall.

**Perbandingan dengan Model Lain**:
- **SelectKBest + SVM**: F1 Score 0.9630, ROC-AUC 0.9987, precision sempurna (1.0), tetapi FN lebih tinggi (3). Tuning hyperparameter (C=1, kernel='linear') meningkatkan performa, menjadikannya sangat kompetitif.
- **RFE + Random Forest**: FN = 3, precision 1.0, F1 Score 0.9630, tetapi recall lebih rendah (0.9286).
- **RFE + SVM**: Performa terlemah (FN = 5, recall 0.8810), kurang ideal karena risiko tinggi kasus *malignant* terlewat.
- **SelectKBest + Random Forest** dan **SelectKBest + Logistic Regression**: FN lebih tinggi (5 dan 4), kurang optimal untuk prioritas deteksi *malignant*.

**Implikasi**: RFE + Logistic Regression unggul dalam meminimalkan FN, yang sangat penting untuk diagnosis kanker payudara guna mencegah kegagalan deteksi kasus *malignant*. SelectKBest + SVM juga sangat andal dengan ROC-AUC tertinggi (0.9987), menunjukkan kemampuan luar biasa dalam membedakan kelas. Seleksi fitur (SelectKBest dan RFE) meningkatkan efisiensi model dengan mengurangi dimensi dan multikolinearitas.

<br>

**Rekomendasi**:
- Terapkan **RFE + Logistic Regression** atau **SelectKBest + SVM** dalam sistem diagnosis rumah sakit untuk mendukung keputusan medis.
- Lakukan validasi silang tambahan untuk memastikan generalisasi model pada data baru.
- Eksplorasi pendekatan deep learning untuk dataset yang lebih besar guna potensi peningkatan performa.
- Pertimbangkan hyperparameter tuning lebih lanjut pada Logistic Regression untuk meningkatkan recall tanpa mengorbankan precision.

<br>

## Referensi
[1] Sung, H., Ferlay, J., Siegel, R. L., Laversanne, M., Soerjomataram, I., Jemal, A., & Bray, F. (2021). Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA: A Cancer Journal for Clinicians, 71(3), 209–249. https://doi.org/10.3322/caac.21660

[2] Kumar, R., & Rani, R. (2019). Feature Selection Method To Improve The Accuracy of Classification. International Journal of Innovative Technology and Exploring Engineering (IJITEE), 8(6), 342–345. Retrieved from https://www.ijitee.org/wp-content/uploads/papers/v8i6/F3421048619.pdf

[3] Kumar, A., & Sharma, R. (2020). Recursive Feature Elimination with Feature Ranking-Based Feature Selection. International Journal of Innovative Research in Technology (IJIRT), 6(2), 223–226. Retrieved from https://ijirt.org/publishedpaper/IJIRT162237_PAPER.pdf

[4] AbdelGawad, A., & Ratner, D. (2007). Adaptive Optimization of Hyperparameters in L2-Regularised Logistic Regression. CS229 Project Report, Stanford University. Retrieved from https://cs229.stanford.edu/proj2007/AbdelGawadRatner-AdaptiveHyperparameterOptimization.pdf

[5] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32. Retrieved from https://doi.org/10.1023/A:1010933404324

[6] Panda, B. S. (n.d.). Support Vector Machines. Indian Institute of Technology Delhi. Retrieved from https://web.iitd.ac.in/~bspanda/SVM.pdf

[7] Dauphin, G. (2019). Confusion Matrix. Laboratoire d'Informatique de Paris Nord. Retrieved from https://www-l2ti.univ-paris13.fr/~dauphin/Confusion_matrix.pdf

