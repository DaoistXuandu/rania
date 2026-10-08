#!/usr/bin/env python3
"""Targeted Laya multilingual verification; standalone, same environment as baseline.

Commands:
    python verification_v1.py
    python verification_v1.py --layers all
    python verification_v1.py --mode verify --candidate torch_mm
    python verification_v1.py --mode verify --candidate my_kernel:matmul
    python verification_v1.py --candidate my_kernel:matmul

Default collect: randomized timing, other_64 anomaly checks, baseline answers,
and FP32 fixtures from layers 0,11,21. --layers all exports all 22 encoder layers.
A fresh collect run overwrites results in the selected output directory.

Candidate interface:
    def matmul(x, w):
        # CPU float32 x=[L,768], w=[768,2304], contiguous.
        # Execute synchronously and return CPU float32 y=[L,2304].
        return ...

The adapter must include real transfers and wait for output before returning.
FPGA/simulator adapters are not supplied. torch_mm is a software sanity check.
FP32 tolerance defaults are not an INT8 accuracy acceptance criterion.
Fixtures may take hundreds of MB; weights are saved once per selected layer.
"""

import argparse
import importlib
import random
import csv
import json
import math
import os
import platform
import resource
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ.pop("LAYA_CPU_AMP", None)
MIB = 1024 ** 2

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this message?",
        "criteria": {
            "billing": "payments, invoices, duplicate charges and refunds",
            "technical": "application crashes, bugs and system errors",
            "other": "anything unrelated to billing or technical problems",
        },
    }
}
BASE_CASES = [
    ("billing", "Saya ditagih dua kali. Tolong kembalikan pembayaran yang duplikat."),
    ("technical", "Aplikasi selalu crash ketika saya membuka pengaturan."),
]


def percentile(values, q):
    """Nearest-rank percentile; works for any configured repeat count."""
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def high_water_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / MIB if sys.platform == "darwin" else value / 1024


class RamSampler:
    def __init__(self, process, interval):
        self.process = process
        self.interval = interval
        self.done = threading.Event()

    def sample(self):
        self.peak = max(self.peak, self.process.memory_info().rss)

    def loop(self):
        while not self.done.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.before = self.process.memory_info().rss
        self.peak = self.before
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join()
        self.sample()
        self.after = self.process.memory_info().rss

    def report(self):
        return {
            "rss_before_mib": self.before / MIB,
            "rss_after_mib": self.after / MIB,
            "sampled_peak_rss_mib": self.peak / MIB,
            "process_lifetime_peak_rss_mib": high_water_mib(),
            "sampling_interval_ms": self.interval * 1000,
        }


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_shapes(value, torch):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): tensor_shapes(v, torch) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_shapes(v, torch) for v in value]
    return None


SCENARIOS = [{'type': 'billing',
  'tokens': 64,
  'text': 'Saya melihat dua tagihan untuk pesanan yang sama, padahal hanya membeli satu kali. '
          'Pembayaran pertama berhasil dan pesanan sudah dikonfirmasi. Beberapa menit kemudian '
          'saldo rekening kembali berkurang dengan nominal yang sama. Mohon periksa kedua '
          'transaksi tersebut, batalkan penagihan kedua, dan kembalikan dana ke rekening asal. '
          'Saya memiliki bukti pembayaran serta nomor pesanan untuk membantu pemeriksaan. Tidak '
          'ada pembelian tambahan yang saya lakukan setelah pesanan pertama selesai.'},
 {'type': 'technical',
  'tokens': 64,
  'text': 'Aplikasi langsung menutup sendiri setiap kali saya membuka halaman pengaturan akun. '
          'Halaman utama masih bisa dibuka, tetapi masalah selalu muncul setelah tombol pengaturan '
          'ditekan. Saya sudah mencoba memulai ulang ponsel dan memperbarui aplikasi, namun '
          'hasilnya tetap sama. Mohon bantu memeriksa penyebab gangguan dan memberikan langkah '
          'perbaikan supaya saya dapat mengubah pengaturan akun. Masalah ini terjadi sejak '
          'pembaruan terakhir dan tidak muncul pada aplikasi lain di perangkat saya.'},
 {'type': 'other',
  'tokens': 64,
  'text': 'Saya ingin mengetahui jam operasional kantor layanan pada hari Sabtu dan apakah '
          'pengunjung perlu membuat janji terlebih dahulu. Rencananya saya datang untuk meminta '
          'informasi tentang program pelatihan yang tersedia bulan depan. Mohon kirimkan alamat '
          'kantor, petunjuk menuju lokasi, dan daftar dokumen yang perlu dibawa. Saya belum '
          'melakukan pembelian atau pembayaran apa pun dan saat ini hanya membutuhkan informasi '
          'mengenai layanan serta jadwal kunjungan.'},
 {'type': 'billing',
  'tokens': 128,
  'text': 'Langganan saya sudah dibatalkan sebelum tanggal perpanjangan, tetapi kartu saya tetap '
          'dikenai biaya untuk periode berikutnya. Saya meminta pengembalian biaya tersebut dan '
          'konfirmasi bahwa penagihan otomatis sudah dihentikan. Pembatalan dilakukan melalui '
          'halaman akun, kemudian saya menerima surel yang menyatakan bahwa paket tidak akan '
          'diperpanjang. Tanggal pada surel tersebut lebih awal daripada tanggal transaksi yang '
          'sekarang muncul pada laporan kartu. Saya tidak memilih paket baru dan tidak memberikan '
          'persetujuan untuk mengaktifkan kembali langganan. Nominal transaksi sesuai dengan harga '
          'paket bulanan sebelumnya. Saya sudah menyimpan bukti pembatalan, nomor pelanggan, dan '
          'rincian pembayaran. Mohon cocokkan waktu pembatalan dengan catatan penagihan agar dana '
          'dapat dikembalikan melalui metode pembayaran semula. Apabila ada biaya administrasi, '
          'saya ingin mendapat penjelasan tertulis beserta perhitungannya. Tolong berikan pula '
          'perkiraan waktu penyelesaian dan nomor laporan agar saya dapat mengikuti status '
          'permintaan ini tanpa mengirim laporan yang sama berulang kali.'},
 {'type': 'technical',
  'tokens': 128,
  'text': 'Saya tidak dapat masuk ke akun karena halaman autentikasi terus kembali ke formulir '
          'awal setelah kata sandi dimasukkan. Mohon bantu memperbaiki masalah login tersebut. '
          'Kata sandi sudah saya atur ulang melalui tautan resmi dan perubahan berhasil '
          'dikonfirmasi. Saat kata sandi baru digunakan, tidak ada pemberitahuan bahwa kredensial '
          'salah, tetapi halaman hanya memuat sebentar lalu meminta saya mengisi data lagi. '
          'Masalah yang sama terjadi pada dua peramban dan juga setelah cache dibersihkan. Koneksi '
          'internet normal dan situs lain dapat dibuka. Saya menggunakan perangkat yang sebelumnya '
          'rutin dipakai untuk mengakses akun ini. Tidak ada perubahan alamat surel maupun nomor '
          'telepon. Saya sudah mencatat waktu kejadian dan mengambil tangkapan layar sebelum '
          'halaman kembali ke formulir. Jika diperlukan, saya dapat memberikan versi peramban '
          'serta sistem operasi yang digunakan. Tolong periksa alur autentikasi dan berikan '
          'langkah pemulihan akses tanpa meminta saya membuat akun baru, karena dokumen pekerjaan '
          'masih tersimpan dalam akun yang lama.'},
 {'type': 'other',
  'tokens': 128,
  'text': 'Saya ingin meminta informasi mengenai kerja sama penyelenggaraan pelatihan untuk '
          'komunitas kami. Mohon arahkan permintaan ini kepada tim yang menangani program '
          'kemitraan. Peserta yang kami rencanakan berjumlah sekitar dua puluh orang dengan latar '
          'belakang usaha kecil. Topik yang dibutuhkan adalah pengenalan layanan, pengelolaan '
          'dokumen, dan prosedur pendaftaran. Kami belum menentukan tanggal karena ingin '
          'menyesuaikan jadwal narasumber. Kegiatan dapat dilaksanakan secara daring atau di ruang '
          'pertemuan komunitas, bergantung pada ketersediaan tim. Saya ingin mengetahui apakah ada '
          'persyaratan jumlah peserta, batas usia, atau dokumen pengajuan yang harus dipenuhi. '
          'Jika tersedia, mohon kirimkan profil program dan kontak penanggung jawab. Kami juga '
          'membutuhkan penjelasan mengenai durasi sesi dan apakah materi dapat dibagikan kepada '
          'peserta setelah kegiatan. Permintaan ini masih pada tahap penjajakan, sehingga belum '
          'ada pemesanan, transaksi, atau kontrak yang dibuat. Saya dapat mengirimkan profil '
          'komunitas dan usulan agenda setelah mendapat petunjuk mengenai prosedur pengajuan yang '
          'sesuai.'},
 {'type': 'billing',
  'tokens': 256,
  'text': 'Nominal pada faktur bulan ini tidak sesuai dengan kesepakatan harga dalam kontrak. Saya '
          'meminta pemeriksaan rincian biaya dan penerbitan faktur yang sudah diperbaiki sebelum '
          'pembayaran dilakukan. Paket yang kami gunakan memiliki tarif tetap dengan jumlah '
          'pengguna yang telah disepakati. Namun faktur terbaru mencantumkan tambahan biaya '
          'pengguna dan layanan pendamping yang tidak pernah kami pesan. Tidak ada perubahan paket '
          'yang diajukan oleh pengelola akun perusahaan. Kami juga tidak menerima pemberitahuan '
          'atau persetujuan tertulis mengenai penambahan layanan. Saya sudah membandingkan faktur '
          'terbaru dengan faktur dua bulan sebelumnya. Biaya utama tetap sama, sementara selisih '
          'muncul pada dua baris tambahan yang belum memiliki penjelasan. Nama perusahaan dan '
          'nomor pelanggan pada faktur sudah benar, sehingga dokumen tersebut memang ditujukan '
          'kepada akun kami. Bagian keuangan belum memproses pembayaran karena perlu memastikan '
          'bahwa nominal sesuai dengan dokumen kontrak. Mohon periksa tanggal mulai layanan '
          'tambahan, siapa yang menyetujuinya, dan dasar perhitungan jumlah pengguna. Apabila '
          'pencatatan tersebut merupakan kesalahan, tolong hapus biaya tambahan dan kirim faktur '
          'pengganti dengan nomor referensi yang dapat ditelusuri. Kami membutuhkan dokumen '
          'koreksi untuk melengkapi arsip pembayaran dan proses persetujuan internal. Jika '
          'ternyata ada perubahan tarif yang berlaku, mohon sertakan pemberitahuan sebelumnya '
          'serta bagian kontrak yang menjadi dasarnya. Saya dapat mengirim salinan kontrak, faktur '
          'lama, dan daftar pengguna aktif. Tolong berikan nomor laporan serta perkiraan waktu '
          'penyelesaian. Kami berharap pemeriksaan dapat selesai sebelum batas pembayaran agar '
          'akun tidak terkena denda atau pembatasan layanan akibat selisih tagihan yang masih '
          'diperselisihkan. Setelah koreksi diterima, bagian keuangan akan memproses pembayaran '
          'sesuai prosedur perusahaan. Kami juga meminta konfirmasi bahwa biaya yang sedang '
          'diperiksa tidak akan ditagihkan kembali pada periode berikutnya.'},
 {'type': 'technical',
  'tokens': 256,
  'text': 'Dokumen yang diunggah melalui portal tidak muncul dalam daftar berkas meskipun '
          'indikator unggahan menyatakan selesai. Saya membutuhkan bantuan untuk memeriksa proses '
          'penyimpanan dan memperbaiki gangguan tersebut. Kejadian ini berlangsung saat saya '
          'mengirim beberapa berkas PDF untuk melengkapi pengajuan. Setiap berkas berukuran kecil '
          'dan sesuai dengan jenis dokumen yang diperbolehkan. Setelah memilih berkas, indikator '
          'kemajuan mencapai akhir dan halaman menampilkan pemberitahuan berhasil. Namun ketika '
          'daftar dokumen dimuat kembali, nama berkas tidak ditemukan. Saya sudah mencoba keluar '
          'lalu masuk kembali, tetapi daftar tetap tidak berubah. Mengunggah satu berkas saja '
          'menghasilkan masalah yang sama. Tidak ada pesan yang menyatakan kuota penyimpanan habis '
          'atau format ditolak. Saya memeriksa koneksi internet dan mencoba jaringan lain untuk '
          'memastikan gangguan bukan berasal dari sambungan lokal. Rekan saya dapat membuka '
          'portal, tetapi pengunggahan pada akun saya tetap tidak tersimpan. Saya belum mengulangi '
          'unggahan berkali-kali karena khawatir berkas sebenarnya tersimpan di belakang layar dan '
          'akan menjadi duplikat. Mohon periksa log permintaan pada waktu kejadian, status '
          'penyimpanan berkas, serta hubungan berkas dengan nomor pengajuan. Jika berkas sudah '
          'tersimpan, tolong pulihkan tampilannya pada daftar dokumen. Jika penyimpanan gagal, '
          'mohon berikan langkah pengunggahan ulang yang aman. Saya memiliki tangkapan layar '
          'pemberitahuan berhasil dan dapat menyampaikan waktu kejadian secara rinci. Dokumen ini '
          'dibutuhkan untuk pekerjaan yang memiliki batas waktu, sehingga saya berharap mendapat '
          'perkiraan penyelesaian dan alternatif pengiriman sementara. Tolong pastikan pula apakah '
          'masalah serupa dapat memengaruhi dokumen lain yang sebelumnya telah diunggah. Setelah '
          'perbaikan, saya ingin memverifikasi bahwa berkas dapat dibuka kembali dan tercatat pada '
          'pengajuan yang benar.'},
 {'type': 'other',
  'tokens': 256,
  'text': 'Saya ingin mengajukan permintaan perubahan alamat korespondensi perusahaan dan '
          'mengetahui dokumen administrasi yang diperlukan. Mohon arahkan saya kepada petugas yang '
          'menangani pembaruan data pelanggan. Kantor kami akan berpindah lokasi pada awal bulan '
          'depan, sementara nama badan usaha dan pengurus tetap sama. Kami ingin memastikan bahwa '
          'surat pemberitahuan serta dokumen resmi berikutnya dikirim ke alamat baru. Saya belum '
          'mengirim formulir karena tidak mengetahui apakah pembaruan perlu diajukan melalui '
          'portal atau dapat disampaikan melalui surel resmi. Mohon berikan formulir yang berlaku '
          'dan petunjuk pengisiannya. Kami memiliki surat keterangan alamat baru serta dokumen '
          'perusahaan yang dapat digunakan sebagai bukti. Jika diperlukan surat kuasa, tolong '
          'jelaskan siapa yang harus menandatangani dan apakah salinan elektronik diterima. Saya '
          'juga ingin mengetahui apakah perubahan alamat dapat dijadwalkan mulai tanggal tertentu '
          'agar surat yang sedang diproses tetap sampai ke kantor lama. Petugas penerima surat '
          'masih berada di lokasi lama selama masa perpindahan. Nomor telepon perusahaan dan '
          'alamat surel utama tidak berubah. Untuk keperluan arsip, kami membutuhkan konfirmasi '
          'tertulis setelah pembaruan selesai. Mohon jelaskan perkiraan lama pemeriksaan dan cara '
          'mengikuti status permintaan tanpa membuat pengajuan baru. Apabila perubahan ini perlu '
          'dilaporkan kepada beberapa bagian secara terpisah, tolong sebutkan bagian yang dimaksud '
          'beserta kontaknya. Kami ingin menghindari perbedaan alamat antarcatatan perusahaan. '
          'Permintaan ini hanya berkaitan dengan pembaruan data administrasi, bukan perubahan '
          'paket layanan. Saya dapat mengirimkan nomor pelanggan melalui kanal resmi setelah '
          'mendapat petunjuk. Tolong informasikan juga apakah dokumen asli perlu dibawa saat '
          'kunjungan atau cukup dilampirkan dalam bentuk salinan. Kami akan menyiapkan berkas '
          'sesuai persyaratan agar proses pemeriksaan dapat dilakukan sekaligus.'},
 {'type': 'billing',
  'tokens': 512,
  'text': 'Pengembalian dana untuk pesanan yang dibatalkan belum masuk ke rekening saya meskipun '
          'pemberitahuan menyebutkan proses telah selesai. Saya meminta penelusuran transaksi '
          'refund dan penjelasan mengenai status dana yang seharusnya dikembalikan. Pesanan '
          'dibatalkan setelah pihak penyedia mengonfirmasi bahwa barang tidak tersedia. Pembatalan '
          'tersebut disetujui dan saya menerima rincian nominal pengembalian melalui surel. Metode '
          'pembayaran awal adalah transfer dari rekening pribadi. Saya tidak meminta pengembalian '
          'dalam bentuk saldo akun atau kupon belanja. Nominal yang disetujui sama dengan '
          'pembayaran awal setelah biaya pengiriman ikut dibatalkan. Dalam pemberitahuan terakhir '
          'terdapat nomor referensi, tetapi tidak ada nama bank tujuan atau tanggal dana dikirim. '
          'Saya sudah memeriksa mutasi rekening sejak tanggal persetujuan pembatalan sampai hari '
          'ini. Tidak ada transaksi masuk dengan nominal yang sesuai. Bank juga meminta bukti '
          'transfer pengembalian agar mereka dapat membantu penelusuran. Karena itu, mohon '
          'kirimkan tanggal pemrosesan, rekening tujuan yang disamarkan, nama bank pengirim, dan '
          'nomor referensi yang dapat digunakan untuk pemeriksaan. Saya ingin memastikan bahwa '
          'rekening tujuan sesuai dengan rekening yang digunakan saat membayar. Pada halaman '
          'pesanan, status pembatalan sudah benar dan tidak ada kewajiban pembayaran tambahan. '
          'Namun status pengembalian dana hanya tertulis selesai, tanpa rincian pelaksanaannya. '
          'Saya telah menghubungi layanan pelanggan sebelumnya dan diminta menunggu beberapa hari '
          'kerja. Masa tunggu tersebut sudah terlewati. Dalam percakapan berikutnya, petugas '
          'kembali memberikan jawaban yang sama tanpa memeriksa nomor laporan sebelumnya. Mohon '
          'gunakan laporan yang sudah ada agar riwayat pemeriksaan tidak terpisah. Saya dapat '
          'melampirkan bukti pembayaran awal, surel persetujuan pembatalan, dan mutasi rekening '
          'yang relevan. Dokumen rekening akan saya kirim melalui kanal resmi sesuai petunjuk '
          'untuk menjaga kerahasiaan data. Apabila dana belum benar-benar dikirim, tolong jelaskan '
          'hambatan pemrosesan dan berikan tanggal penyelesaian yang dapat diikuti. Jika transaksi '
          'pengembalian ditolak oleh bank, mohon sampaikan alasannya dan langkah koreksi yang '
          'diperlukan. Saya bersedia memverifikasi nama pemilik rekening, tetapi tidak ingin '
          'mengubah tujuan pengembalian ke rekening pihak lain. Tolong pastikan bahwa proses tidak '
          'menghasilkan pengembalian ganda apabila transaksi lama ternyata masih menunggu '
          'penyelesaian. Saya membutuhkan konfirmasi status yang didukung catatan transaksi, bukan '
          'sekadar pemberitahuan umum bahwa pengajuan diterima. Dana tersebut akan digunakan '
          'kembali untuk kebutuhan lain, sehingga ketidakjelasan jadwal menyulitkan perencanaan '
          'saya. Setelah penelusuran selesai, mohon kirimkan ringkasan hasil pemeriksaan dan bukti '
          'pengiriman dana. Jika nominal yang dikembalikan berbeda dari persetujuan semula, '
          'sertakan rincian potongan beserta dasar perhitungannya. Saya juga meminta agar pesanan '
          'yang telah dibatalkan tidak menghasilkan tagihan baru. Tidak ada pesanan pengganti yang '
          'saya setujui dan tidak ada barang yang saya terima. Seluruh permintaan ini berkaitan '
          'dengan penyelesaian pengembalian pembayaran untuk pesanan tersebut. Mohon arahkan '
          'laporan kepada tim pembayaran yang dapat memeriksa transaksi refund secara langsung dan '
          'memberikan nomor pelacakan yang benar.'},
 {'type': 'technical',
  'tokens': 512,
  'text': 'Hasil ekspor laporan dari aplikasi kehilangan sebagian baris data meskipun tabel di '
          'layar menampilkan seluruh catatan. Saya meminta pemeriksaan fungsi ekspor dan perbaikan '
          'agar dokumen yang diunduh memuat data lengkap. Masalah ditemukan ketika tim kami '
          'menyiapkan laporan kegiatan mingguan. Pada halaman daftar, jumlah catatan sesuai dengan '
          'data yang dimasukkan oleh operator. Filter tanggal sudah dipilih untuk seluruh minggu '
          'dan tidak ada pembatasan berdasarkan petugas. Ketika tombol ekspor digunakan, aplikasi '
          'menghasilkan berkas tanpa menampilkan pesan kesalahan. Namun jumlah baris di dalam '
          'berkas lebih sedikit daripada jumlah yang terlihat pada layar. Beberapa catatan yang '
          'hilang berasal dari hari yang berbeda, sehingga masalah tidak tampak terbatas pada satu '
          'tanggal. Saya sudah mengulangi proses dengan filter yang lebih sempit dan menemukan '
          'bahwa sebagian catatan dapat muncul jika diekspor secara terpisah. Ini membuat kami '
          'menduga ada masalah pada pengambilan data atau batas jumlah hasil dalam satu '
          'permintaan. Saya belum dapat memastikan penyebabnya dan berharap tim teknis memeriksa '
          'proses tersebut. Ekspor ke format spreadsheet maupun PDF menunjukkan jumlah catatan '
          'yang tidak lengkap. Nama kolom dan format tanggal masih benar, tetapi beberapa baris '
          'tidak disertakan. Data pada aplikasi sendiri tidak terlihat terhapus. Rekan yang '
          'memiliki izin akses sama mencoba proses dari perangkat lain dan mendapatkan hasil '
          'serupa. Kami menggunakan versi aplikasi terbaru dan telah mencoba peramban yang '
          'berbeda. Membersihkan cache tidak mengubah hasil. Saya mencatat jumlah baris yang '
          'terlihat di halaman, jumlah baris hasil ekspor, rentang tanggal, dan waktu setiap '
          'percobaan. Catatan tersebut dapat diberikan sebagai bahan reproduksi masalah. Saya juga '
          'menyiapkan contoh nomor catatan yang hilang agar tim dapat membandingkan hasil '
          'permintaan dengan data yang tersimpan. Untuk menjaga informasi internal, contoh berkas '
          'akan dikirim melalui kanal dukungan resmi. Mohon jelaskan apakah fungsi ekspor memiliki '
          'batas jumlah baris yang tidak ditampilkan pada antarmuka. Jika memang ada batas, kami '
          'membutuhkan petunjuk untuk mengambil seluruh data secara aman. Jika masalah berasal '
          'dari kesalahan implementasi, tolong berikan perkiraan waktu perbaikan dan cara '
          'sementara yang dapat digunakan tanpa mengubah data sumber. Kami berharap tidak perlu '
          'menyalin tabel secara manual karena cara itu mudah menghasilkan kesalahan. Laporan '
          'diperlukan oleh beberapa tim, sehingga kelengkapan catatan harus dapat diverifikasi. '
          'Setelah perbaikan, kami akan membandingkan jumlah baris dan nomor catatan dengan '
          'tampilan aplikasi. Tolong pastikan urutan serta isi kolom juga tetap konsisten. Saya '
          'meminta nomor laporan agar komunikasi berikutnya dapat mengikuti riwayat yang sama. '
          'Jika dibutuhkan sesi pemeriksaan bersama, saya dapat menjadwalkan demonstrasi pada akun '
          'uji yang memiliki pola data serupa. Tim kami tidak ingin menghapus atau mengunggah '
          'ulang catatan sebelum mengetahui penyebab masalah, karena tindakan tersebut dapat '
          'merusak riwayat pekerjaan. Mohon konfirmasikan apakah data asli tetap aman dan apakah '
          'pengguna lain berpotensi mengalami gangguan serupa. Permintaan utama kami adalah '
          'memperbaiki ekspor yang tidak lengkap, mempertahankan data yang ada, dan memperoleh '
          'hasil unduhan yang sesuai dengan catatan pada aplikasi.'},
 {'type': 'other',
  'tokens': 512,
  'text': 'Saya ingin meminta informasi mengenai prosedur kunjungan edukasi untuk rombongan '
          'mahasiswa ke fasilitas perusahaan. Mohon arahkan permintaan ini kepada tim yang '
          'menangani hubungan masyarakat atau kegiatan kunjungan institusi. Tujuan kegiatan adalah '
          'mengenalkan proses kerja dan pengelolaan dokumen kepada mahasiswa yang sedang mengikuti '
          'mata kuliah administrasi. Kami belum melakukan pemesanan dan masih menyesuaikan rencana '
          'dengan kebijakan penerimaan pengunjung. Jumlah peserta diperkirakan tiga puluh orang, '
          'didampingi dua dosen. Kami dapat membagi rombongan menjadi kelompok kecil apabila '
          'jumlah tersebut melebihi kapasitas satu sesi. Waktu kunjungan yang diusulkan berada '
          'pada hari kerja bulan depan, tetapi tanggalnya masih dapat disesuaikan dengan jadwal '
          'perusahaan. Mohon jelaskan hari yang tersedia, durasi kegiatan, dan batas jumlah '
          'peserta. Kami juga ingin mengetahui apakah diperlukan surat permohonan resmi dari '
          'fakultas. Jika ada format surat tertentu, tolong kirimkan contoh atau daftar informasi '
          'yang wajib dicantumkan. Kami dapat menyiapkan identitas penanggung jawab, daftar '
          'peserta, dan tujuan pembelajaran. Untuk persiapan keberangkatan, saya membutuhkan '
          'alamat lokasi serta petunjuk mengenai pintu masuk yang digunakan oleh rombongan. Mohon '
          'sampaikan apakah kendaraan kampus dapat berhenti di area penerimaan dan apakah tersedia '
          'tempat parkir untuk bus kecil. Jika peserta perlu berjalan dari lokasi parkir, kami '
          'ingin memperkirakan waktu kedatangan agar tidak terlambat. Salah satu peserta '
          'membutuhkan akses yang mudah dilalui, sehingga informasi mengenai tangga dan jalur '
          'masuk akan membantu persiapan. Kami ingin mengikuti seluruh ketentuan keamanan yang '
          'berlaku. Tolong jelaskan dokumen identitas yang perlu dibawa, aturan pakaian, barang '
          'yang tidak boleh masuk, dan kebijakan penggunaan kamera. Jika pengambilan gambar hanya '
          'diperbolehkan pada area tertentu, kami akan menyampaikan aturan tersebut kepada peserta '
          'sebelum berangkat. Kami juga dapat mengumpulkan pertanyaan mahasiswa terlebih dahulu '
          'agar sesi diskusi lebih terarah. Topik yang ingin dipahami meliputi pembagian tugas, '
          'alur pemeriksaan dokumen, dan cara perusahaan menjaga mutu layanan. Kami tidak meminta '
          'akses ke informasi rahasia atau area yang tidak diperbolehkan untuk pengunjung. Apabila '
          'ada materi pengantar yang dapat dibaca sebelum kunjungan, mohon bagikan tautan atau '
          'dokumen resminya. Dosen pendamping akan menggunakan materi tersebut untuk mempersiapkan '
          'peserta. Kami juga ingin mengetahui apakah tersedia narasumber untuk penjelasan singkat '
          'dan apakah diskusi dilakukan sebelum atau sesudah tur. Untuk administrasi kampus, kami '
          'memerlukan konfirmasi tertulis mengenai jadwal yang telah disepakati. Jika tanggal '
          'belum dapat ditentukan sekarang, cukup berikan prosedur pengajuan serta perkiraan waktu '
          'tanggapan. Mohon sebutkan kontak yang dapat dihubungi agar koordinasi tidak tersebar ke '
          'banyak bagian. Saya akan menjadi penghubung utama dan mengumpulkan seluruh dokumen dari '
          'kampus. Apabila kunjungan langsung belum tersedia, kami bersedia mempertimbangkan sesi '
          'pengenalan daring dengan tujuan pembelajaran yang sama. Kami berharap mendapat '
          'penjelasan mengenai pilihan kegiatan dan persyaratannya. Permintaan ini merupakan '
          'permohonan informasi kunjungan institusi, sehingga kami membutuhkan arahan mengenai '
          'jadwal, dokumen, dan tata tertib yang harus dipenuhi sebelum mengajukan surat resmi.'}]


def make_cases(tokenizer, targets):
    """Twelve authored scenarios; crop each to its requested state-token length."""
    supported = {64, 128, 256, 512}
    if not set(targets).issubset(supported):
        raise ValueError("Profiling v2 menyediakan panjang 64, 128, 256, 512 saja")
    cases = []
    for scenario in SCENARIOS:
        target = scenario["tokens"]
        if target not in targets:
            continue
        ids = tokenizer.encode(scenario["text"], add_special_tokens=False)
        if len(ids) < target:
            raise ValueError(f"Draft {scenario['type']}_{target} hanya {len(ids)} token; "
                             "jangan menambahkan filler otomatis. Perlu perluasan draft.")
        # Decode/encode may shift a boundary: search nearby cuts for an exact count.
        text = None
        for cut in [target] + [target + d for i in range(1, 33) for d in (-i, i)]:
            if not 1 <= cut <= len(ids):
                continue
            candidate = tokenizer.decode(ids[:cut], skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
            actual = len(tokenizer.encode(candidate, add_special_tokens=False))
            if actual == target:
                text = candidate
                break
        if text is None:
            raise ValueError(f"Tidak menemukan tepat {target} token setelah round-trip tokenizer "
                             f"untuk {scenario['type']}; sesuaikan batas akhir draft.")
        cases.append({"id": f"{scenario['type']}_{target}", "text": text,
                      "expected": scenario["type"], "target_state_tokens": target,
                      "actual_state_tokens": target,
                      "source_draft_tokens": len(ids)})
    return cases


def install_module_scopes(model, record_function):
    """Instrument all modules so functional operations retain their parent layer."""
    handles = []
    stacks = defaultdict(list)

    def pre(label):
        def hook(module, args):
            scope = record_function(label)
            scope.__enter__()
            stacks[id(module)].append(scope)
        return hook

    def post(module, args, output):
        stack = stacks[id(module)]
        if stack:
            stack.pop().__exit__(None, None, None)

    try:
        for name, module in model.named_modules():
            label = "MODULE::" + (name or "<model>")
            handles.append(module.register_forward_pre_hook(pre(label)))
            handles.append(module.register_forward_hook(post, always_call=True))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles


def operation_rows(prof, case_id, repeats):
    events = [e for e in prof.events() if e.name.startswith("aten::")]
    total = sum(e.self_cpu_time_total for e in events)
    groups = defaultdict(lambda: [0.0, 0, 0])
    for event in events:
        parent = event.cpu_parent
        layer = "<outside model scopes>"
        while parent is not None:
            if parent.name.startswith("MODULE::"):
                layer = parent.name.removeprefix("MODULE::")
                break
            parent = parent.cpu_parent
        shape = json.dumps(event.input_shapes)
        key = (layer, event.name, shape)
        group = groups[key]
        group[0] += event.self_cpu_time_total
        group[1] += 1
        group[2] += event.self_cpu_memory_usage
    rows = []
    for (layer, op, shape), (us, count, memory) in groups.items():
        rows.append({"case": case_id, "module": layer, "operation": op,
                     "input_shapes": shape, "calls": count,
                     "calls_per_inference": count / repeats,
                     "self_cpu_ms": us / 1000,
                     "self_cpu_ms_per_inference": us / 1000 / repeats,
                     "share_of_aten_self_cpu_pct": us / total * 100 if total else 0,
                     "net_self_tensor_allocation_bytes": memory})
    return sorted(rows, key=lambda x: x["self_cpu_ms"], reverse=True)


def resolve_kernel(spec, torch):
    # A torch-only implementation is provided to check adapter plumbing.
    if spec == 'torch_mm':
        return lambda x, w: torch.mm(x, w)
    module_name, separator, function_name = spec.partition(':')
    if not separator:
        raise ValueError('--candidate harus torch_mm atau nama_module:nama_function')
    return getattr(importlib.import_module(module_name), function_name)


def checked_output(kernel, x, w, torch):
    # Contract: x=[L,768], w=[768,2304], CPU float32 -> y=[L,2304].
    # The candidate adapter must wait for completion and return host output.
    y = kernel(x, w)
    if not isinstance(y, torch.Tensor):
        raise TypeError('Kernel harus mengembalikan torch.Tensor, bukan future/handle')
    if y.device.type != 'cpu' or y.dtype != torch.float32:
        raise TypeError('Keluaran adapter harus CPU float32 (dequantize bila INT8)')
    if tuple(y.shape) != (x.shape[0], w.shape[1]):
        raise ValueError(f'Shape keluaran salah: {tuple(y.shape)}')
    return y


def verify_fixtures(out, kernel, name, torch, atol, rtol):
    manifest = json.loads((out / 'fixtures' / 'manifest.json').read_text())
    cache, rows = {}, []
    for item in manifest:
        if item['weight_file'] not in cache:
            cache[item['weight_file']] = torch.load(
                out / 'fixtures' / item['weight_file'], map_location='cpu', weights_only=True)
        weight = cache[item['weight_file']]
        data = torch.load(out / 'fixtures' / item['fixture_file'],
                          map_location='cpu', weights_only=True)
        x, reference = data['x'], data['y']
        with torch.inference_mode():
            actual = checked_output(kernel, x, weight, torch)
            difference = actual - reference
            passed = torch.allclose(actual, reference, atol=atol, rtol=rtol)
            rows.append({'case': item['case'], 'module': item['module'],
                         'kernel': name, 'passed': passed,
                         'max_abs_error': difference.abs().max().item(),
                         'rmse': difference.square().mean().sqrt().item(),
                         'atol': atol, 'rtol': rtol})
    write_csv(out / 'kernel_checks.csv', rows)
    passed = sum(row['passed'] for row in rows)
    print(f'Pemeriksaan {name}: {passed}/{len(rows)} tensor PASS', flush=True)
    return passed == len(rows)


def main():
    parser = argparse.ArgumentParser(description='Verifikasi terarah Laya multilingual CPU FP32')
    parser.add_argument('--mode', choices=['collect', 'verify'], default='collect')
    parser.add_argument('--output', default='verification_v1_results')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10,
                        help='Pengulangan tiap kasus per ronde; total default 30')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--layers', default='0,11,21',
                        help='Layer yang diekspor/diganti; gunakan all untuk seluruh 22 layer')
    parser.add_argument('--candidate', help='torch_mm atau module:function')
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-4)
    args = parser.parse_args()
    if min(args.rounds, args.repeats, args.warmup, args.threads) < 1:
        parser.error('Jumlah pengulangan dan threads harus positif')
    if args.atol < 0 or args.rtol < 0:
        parser.error('Toleransi tidak boleh negatif')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(args.threads)
    if args.mode == 'verify':
        if not args.candidate:
            parser.error('--mode verify memerlukan --candidate')
        kernel = resolve_kernel(args.candidate, torch)
        ok = verify_fixtures(out, kernel, args.candidate, torch, args.atol, args.rtol)
        return 0 if ok else 2
    import laya
    from torch.profiler import profile, ProfilerActivity, record_function
    print('Memuat multilingual CPU FP32...', flush=True)
    agent = laya.load('convaiinnovations/laya', subfolder='multilingual', device='cpu')
    agent.model.eval()
    if any(p.dtype != torch.float32 or p.device.type != 'cpu'
           for p in agent.model.parameters()):
        raise RuntimeError('Baseline harus CPU FP32')
    cases = make_cases(agent.tok, [64,128,256,512])
    def predict(case):
        return agent.predict(case['text'], QUESTIONS, max_len=1024)
    modules = dict(agent.model.named_modules())
    layer_ids = list(range(22)) if args.layers == 'all' else [int(n) for n in args.layers.split(',')]
    targets = [f'encoder.layers.{i}.{suffix}' for i in layer_ids
               for suffix in ['attn.Wqkv', 'mlp.Wi']]
    for name in targets:
        if name not in modules or tuple(modules[name].weight.shape) != (2304,768):
            raise ValueError(f'Layer tidak ditemukan atau shape berbeda: {name}')
        if modules[name].bias is not None:
            raise ValueError(f'Layer {name} memiliki bias; adapter v1 untuk layer tanpa bias')
    report = {'config': vars(args), 'versions': {'torch': torch.__version__,
              'laya': getattr(laya, '__version__', 'unknown')}, 'platform': platform.platform(),
              'selected_modules': targets, 'cases': cases,
              'notes': [
                  'Randomized order reduces order bias; separate runs still vary with system conditions.',
                  'FP32 tensor tolerances are configurable and are not INT8 acceptance criteria.',
                  'torch_mm is a software sanity check, not an FPGA implementation.',
                  'End-to-end time includes adapter transfers only if the candidate actually performs them.',
                  'Candidate must synchronously return host CPU float32 output.',
                  'Only selected layers are replaced; default is 0,11,21, not all 22 layers.',
                  'Baseline mistakes are recorded, not relabelled or used to train the model.',
              ]}
    answers = {}
    print('Warm-up dan pencatatan keputusan baseline...', flush=True)
    for case in cases:
        for _ in range(args.warmup):
            result = predict(case)
        answer = result['answers']['department']
        answers[case['id']] = answer
        case['baseline_answer'] = answer
        case['baseline_usage'] = result.get('usage')
        print(f"{case['id']}: target={case['expected']} prediksi={answer['choice']}")

    # Every round contains every case equally often; case occurrences are shuffled.
    rng = random.Random(args.seed)
    timings = []
    print('Pengukuran baseline dengan urutan acak...', flush=True)
    for round_id in range(args.rounds):
        order = [case for case in cases for _ in range(args.repeats)]
        rng.shuffle(order)
        for position, case in enumerate(order):
            start = time.perf_counter()
            result = predict(case)
            ms = (time.perf_counter() - start) * 1000
            timings.append({'round': round_id+1, 'position': position+1,
                            'case': case['id'], 'wall_ms': ms,
                            'prediction': result['answers']['department']['choice']})
        write_csv(out / 'randomized_latency.csv', timings)
        print(f'Ronde {round_id+1}/{args.rounds} selesai', flush=True)
    summary = []
    for case in cases:
        values = [r['wall_ms'] for r in timings if r['case'] == case['id']]
        row = {'case': case['id'], 'median_ms': statistics.median(values),
               'p95_ms': percentile(values,.95), 'min_ms': min(values), 'max_ms': max(values)}
        summary.append(row)
    report['randomized_latency'] = summary
    write_csv(out / 'randomized_summary.csv', summary)

    (out / 'verification_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,
                                                            default=str),encoding='utf-8')
    (out / 'baseline_answers.json').write_text(json.dumps(answers,indent=2,ensure_ascii=False),encoding='utf-8')
    # Re-profile other_64 in several separate sessions, using per-event timings.
    print('Memeriksa ulang anomali other_64...', flush=True)
    case = next(c for c in cases if c['id'] == 'other_64')
    anomaly = []
    for session in range(3):
        for _ in range(args.warmup):
            predict(case)
        handles = install_module_scopes(agent.model, record_function)
        try:
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                for _ in range(5):
                    predict(case)
        finally:
            for handle in handles:
                handle.remove()
        grouped = defaultdict(list)
        for event in prof.events():
            if event.name != 'aten::mm':
                continue
            parent = event.cpu_parent
            while parent is not None and not parent.name.startswith('MODULE::'):
                parent = parent.cpu_parent
            if parent is not None and parent.name.endswith('.attn.Wqkv'):
                grouped[parent.name.removeprefix('MODULE::')].append(event.self_cpu_time_total/1000)
        for name, values in grouped.items():
            anomaly.append({'session': session+1, 'module': name, 'calls': len(values),
                            'median_self_cpu_ms': statistics.median(values),
                            'p95_self_cpu_ms': percentile(values,.95), 'max_self_cpu_ms': max(values)})
        del prof
    write_csv(out / 'other64_qkv_checks.csv', anomaly)

    # Export once per case/layer; weights are saved once, not once per case.
    fixtures_dir = out / 'fixtures'
    fixtures_dir.mkdir(exist_ok=True)
    manifest, weight_paths = [], {}
    for name in targets:
        filename = name.replace('.', '_') + '_weight.pt'
        weight_paths[name] = filename
        torch.save(modules[name].weight.detach().t().contiguous().cpu(), fixtures_dir / filename)
    print(f'Ekspor tensor dari {len(targets)} layer untuk 12 kasus...', flush=True)
    for case in cases:
        handles, captured = [], set()
        def exporter(name):
            def hook(module, inputs, output):
                if name in captured:
                    raise RuntimeError(f'Layer {name} dipanggil lebih dari sekali; perlu ID panggilan')
                captured.add(name)
                x = inputs[0].detach().reshape(-1,768).contiguous().cpu().clone()
                y = output.detach().reshape(-1,2304).contiguous().cpu().clone()
                filename = case['id'] + '__' + name.replace('.', '_') + '.pt'
                torch.save({'x': x, 'y': y}, fixtures_dir / filename)
                manifest.append({'case': case['id'], 'module': name,
                                 'fixture_file': filename, 'weight_file': weight_paths[name],
                                 'x_shape': list(x.shape), 'w_shape': [768,2304],
                                 'y_shape': list(y.shape), 'original_output_shape': list(output.shape)})
            return hook
        try:
            for name in targets:
                handles.append(modules[name].register_forward_hook(exporter(name)))
            predict(case)
        finally:
            for handle in handles:
                handle.remove()
        if captured != set(targets):
            raise RuntimeError('Tidak seluruh layer target dieksekusi; ekspor belum lengkap')
    (fixtures_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # This check verifies tensor layout/export only, NOT hardware correctness.
    reference_ok = verify_fixtures(out, resolve_kernel('torch_mm',torch),
                                   'torch_mm_export_sanity',torch,args.atol,args.rtol)
    report['export_sanity_passed'] = reference_ok
    if not reference_ok:
        raise RuntimeError('Acuan ekspor gagal diperiksa; jangan lanjut mengganti kernel')

    if args.candidate:
        kernel = resolve_kernel(args.candidate, torch)
        candidate_ok = verify_fixtures(out,kernel,args.candidate,torch,args.atol,args.rtol)
        report['candidate_tensor_passed'] = candidate_ok
        if candidate_ok:
            print('Perbandingan inferensi baseline vs candidate...', flush=True)
            originals = {name: modules[name].forward for name in targets}
            weights = {name: modules[name].weight.detach().t().contiguous().cpu() for name in targets}
            def replacement(name):
                def forward(x):
                    shape = tuple(x.shape)
                    y = checked_output(kernel,x.reshape(-1,768).contiguous(),weights[name],torch)
                    return y.reshape(*shape[:-1],2304)
                return forward
            replacements = {name: replacement(name) for name in targets}
            def activate(mode):
                for name in targets:
                    modules[name].forward = replacements[name] if mode == 'candidate' else originals[name]
            e2e = []
            try:
                activate('candidate')
                for case in cases:
                    for _ in range(args.warmup):
                        predict(case)
                for round_id in range(args.rounds):
                    order = [c for c in cases for _ in range(args.repeats)]
                    rng.shuffle(order)
                    for case in order:
                        modes = ['baseline','candidate']
                        rng.shuffle(modes)
                        for mode in modes:
                            activate(mode)  # Switching itself is outside the timed interval.
                            start = time.perf_counter()
                            result = predict(case)
                            ms = (time.perf_counter()-start)*1000
                            answer = result['answers']['department']
                            original = answers[case['id']]
                            keys = original['probabilities']
                            drift = max(abs(answer['probabilities'][k]-original['probabilities'][k]) for k in keys)
                            e2e.append({'round':round_id+1,'case':case['id'],'mode':mode,
                                        'wall_ms':ms,'prediction':answer['choice'],
                                        'baseline_prediction':original['choice'],
                                        'choice_matches_baseline':answer['choice']==original['choice'],
                                        'max_reported_probability_drift':drift})
                    write_csv(out / 'candidate_e2e.csv',e2e)
                    print(f'Candidate ronde {round_id+1}/{args.rounds} selesai',flush=True)
            finally:
                activate('baseline')
            report['candidate_choices_preserved'] = all(r['choice_matches_baseline'] for r in e2e)
            report['candidate_e2e_summary'] = []
            for case in cases:
                row = {'case':case['id']}
                for mode in ['baseline','candidate']:
                    values=[r['wall_ms'] for r in e2e if r['case']==case['id'] and r['mode']==mode]
                    row[mode+'_median_ms']=statistics.median(values)
                    row[mode+'_p95_ms']=percentile(values,.95)
                row['speedup']=row['baseline_median_ms']/row['candidate_median_ms']
                report['candidate_e2e_summary'].append(row)
            write_csv(out / 'candidate_e2e_summary.csv',report['candidate_e2e_summary'])
        else:
            print('Candidate gagal toleransi tensor; penggantian dalam model dilewati.',flush=True)
    (out / 'verification_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,
                                                            default=str),encoding='utf-8')
    (out / 'baseline_answers.json').write_text(json.dumps(answers,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Selesai: {out.resolve()}')
    print('torch_mm hanya pemeriksaan software; angka simulator bukan latency FPGA.')
    return 2 if args.candidate and (not report.get('candidate_tensor_passed') or
                                  not report.get('candidate_choices_preserved')) else 0


if __name__ == '__main__':
    raise SystemExit(main())
