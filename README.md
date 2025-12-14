# 🍛 Klasifikasi Makanan Minangkabau dengan MobileNetV3

![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-FF6F00?logo=tensorflow)
![Railway](https://img.shields.io/badge/Deployed%20on-Railway-0B0D0E?logo=railway)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)

---

## 📋 Deskripsi
Model **MobileNetV3** digunakan untuk klasifikasi gambar makanan khas **Minangkabau** ke dalam **9 kelas** dengan API dibangun menggunakan framework python yaitu **FastAPI** dan dideploy di PaaS yaitu **Railway**.

---

## 🍽️ Kelas Makanan
`ayam_goreng`, `ayam_pop`, `daging_rendang`, `dendeng_batokok`,  
`gulai_ikan`, `gulai_tambusu`, `gulai_tunjang`, `telur_balado`, `telur_dadar`

---

## 🚀 Fitur
- 🔍 Single & Batch Prediction  
- 🔑 API Key Authentication  
- 🩺 Health Monitoring  
- ⚡ Fast Inference (TFLite optimized)  
- 🧾 File Validation  


---

## 🔌 API Endpoint
**Base URL: https://klasifikasi-makanan-minangkabau.up.railway.app**  


| Endpoint | Method | Deskripsi |
|-----------|---------|-----------|
| `/health` | GET | Mengecek status API |
| `/classes` | GET | Mendapatkan daftar kelas makanan |
| `/predict` | POST | Prediksi satu gambar makanan |
| `/batch-predict` | POST | Prediksi beberapa gambar sekaligus |

---

## 🔐 API Key
Untuk akses API ini, silakan **hubungi pengembang proyek** untuk mendapatkan API Key resmi.  
> 📩 *Contact fajaralfad85@gmail.com*

