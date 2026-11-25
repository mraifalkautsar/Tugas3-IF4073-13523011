# Segmentasi Citra untuk Tugas 3 IF4073

Aplikasi MATLAB untuk mengambil objek dengan deteksi tepi, membagi citra dengan *split and merge*, dan memberi label pada tiap piksel dengan model segmentasi semantik CamVid.

## Dependensi

- MATLAB untuk Windows dengan Image Processing Toolbox. Aplikasi diuji pada R2026b.
- Khusus CamVid, MATLAB dengan Java dan Ubuntu di WSL 2 dengan Python 3.12, NumPy, Pillow, PyTorch, serta Torchvision. CamVid dapat dijalankan di CPU atau GPU dengan CUDA.

## Menjalankan aplikasi

Buka folder repositori di MATLAB, lalu jalankan perintah berikut.

```matlab
segmentationApp
```

Klik **Open image**, pilih tab metode, atur parameter, lalu klik **Run**. Tab **CamVid semantic** memerlukan bobot model di `artifacts/best.pt`.

## Menyiapkan CamVid

Jalankan perintah ini dari folder repositori di Ubuntu WSL. Dataset yang digunakan berasal dari [repositori CamVid milik lih627](https://github.com/lih627/CamVid).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy==2.5.4 pillow==12.3.0
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
git clone --depth 1 --filter=blob:none --sparse https://github.com/lih627/CamVid.git data/CamVid
git -C data/CamVid sparse-checkout set CamVid_RGB CamVid_Label
.venv/bin/python -m semantic prepare
.venv/bin/python -m semantic train
```

Untuk instalasi khusus CPU, ganti indeks PyTorch menjadi `https://download.pytorch.org/whl/cpu`. Setelah pelatihan, jalankan `.venv/bin/python -m semantic evaluate` untuk mengevaluasi model pada data uji CamVid.

## Menguji dengan BSDS500

Ekstrak [delapan citra uji tugas](https://drive.google.com/file/d/1VlWheiTmPjIeFGtKfSCxQWu40PWBmxr_/view) ke `data/bsds500_test_8/`. Dari folder repositori di MATLAB, jalankan perintah berikut. Hasil deteksi tepi dan *split and merge* tersimpan di `results/bsds500/`.

```matlab
addpath('evaluation')
evaluateBSDS500
```
