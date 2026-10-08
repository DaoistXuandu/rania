#!/usr/bin/env python3
"""Laya multilingual CPU/FP32 profiling.

Run in the same environment as baseline_multilingual.py:
    python -m pip install psutil
    python profiling_v2.py

Writes v2_summary.json, v2_latency.csv, v2_operators.csv,
v2_layer_operations.csv, v2_modules.csv and one Chrome trace per case into
profiling_v2/.
Twelve authored scenarios measure performance; they are not an accuracy dataset.
RSS peaks are sampled; the OS high-water mark is process-lifetime, not phase-local.
Profiler memory columns are allocation counters, not process peak RAM.
"""

import argparse
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
RESULT_PREFIX = "v2"

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


def result_path(out, name):
    return out / f"{RESULT_PREFIX}_{name}"


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="profiling_v2")
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sample-ms", type=float, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=1024)
    args = parser.parse_args()
    if min(args.repeats, args.profile_repeats, args.warmup, args.threads,
           args.max_len, *args.lengths) < 1 or args.sample_ms <= 0:
        parser.error("Semua jumlah, panjang, dan interval harus positif")
    if max(args.lengths) + 256 > args.max_len:
        parser.error("Sisakan 256 token untuk pertanyaan: lengths + 256 <= max-len")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import psutil
    except ImportError:
        parser.exit(1, "Install dahulu: python -m pip install psutil\n")
    process = psutil.Process()
    interval = args.sample_ms / 1000
    report = {"config": vars(args), "platform": platform.platform(),
              "python": sys.version, "memory": {}, "cases": [],
              "notes": [
                  "Twelve unique authored scenarios, four each for billing, technical and other.",
                  "Texts are cropped to exact state-token targets, without repeated filler.",
                  "These authored examples do not establish real-world accuracy.",
                  "Target length refers to state text; actual packed model shapes are recorded.",
                  "Latency runs have no profiler, module scopes or RAM sampling thread.",
                  "Memory runs are separate; sampled peaks may miss short spikes.",
                  "OS peak RSS is process-lifetime, not phase-local; RSS is not parameter size.",
                  "Operator percentages use summed aten self CPU, not predict wall time.",
                  "Profiler tensor allocation counters are not peak resident memory.",
                  "First inference excludes download/load; load may include cached download checks.",
              ]}
    print("Mengimpor runtime...", flush=True)
    with RamSampler(process, interval) as ram:
        import torch
        import laya
        from torch.profiler import profile, ProfilerActivity, record_function
    report["memory"]["runtime_import"] = ram.report()
    torch.set_num_threads(args.threads)
    report["versions"] = {"torch": torch.__version__,
                          "laya": getattr(laya, "__version__", "unknown"),
                          "psutil": psutil.__version__}
    print("Memuat multilingual CPU FP32...", flush=True)
    with RamSampler(process, interval) as ram:
        start = time.perf_counter()
        agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cpu")
        load_s = time.perf_counter() - start
    agent.model.eval()
    report["memory"]["model_load"] = ram.report()
    parameters = list(agent.model.parameters())
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in parameters):
        raise RuntimeError("Baseline harus seluruhnya CPU FP32")
    report["model"] = {"load_seconds": load_s, "device": str(agent.device),
                       "parameter_count": sum(p.numel() for p in parameters),
                       "parameter_mib": sum(p.numel() * p.element_size() for p in parameters) / MIB,
                       "buffer_mib": sum(b.numel() * b.element_size() for b in agent.model.buffers()) / MIB,
                       "cpu_threads": torch.get_num_threads()}
    def predict(text):
        return agent.predict(text, QUESTIONS, max_len=args.max_len)

    # This is truly the first predict in this process, before case generation/warm-up.
    print("Mengukur inferensi pertama...", flush=True)
    with RamSampler(process, interval) as ram:
        start = time.perf_counter()
        first = predict(BASE_CASES[0][1])
        first_ms = (time.perf_counter() - start) * 1000
    report["memory"]["first_inference"] = ram.report()
    report["first_inference"] = {"wall_ms_with_ram_sampler": first_ms,
                                  "prediction": first["answers"]["department"]["choice"]}
    cases = make_cases(agent.tok, args.lengths)
    result_path(out, "input_cases.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Mengukur {len(cases)} pesan berbeda; v2_input_cases.json menyimpan teks aktual.",
          flush=True)
    latency_rows, all_ops, all_layers = [], [], []
    modules = []
    for name, module in agent.model.named_modules():
        modules.append({"name": name or "<model>", "class": type(module).__name__,
                        "direct_parameter_shapes": json.dumps({
                            n: list(p.shape) for n, p in module.named_parameters(recurse=False)})})
    write_csv(result_path(out, "modules.csv"), modules)

    for case in cases:
        text = case["text"]
        print(f"\n{case['id']}: state={case['actual_state_tokens']} token", flush=True)
        for _ in range(args.warmup):
            predict(text)
        times, correct = [], 0
        for i in range(args.repeats):
            start = time.perf_counter()
            result = predict(text)
            ms = (time.perf_counter() - start) * 1000
            prediction = result["answers"]["department"]["choice"]
            correct += int(prediction == case["expected"])
            times.append(ms)
            latency_rows.append({"case": case["id"], "iteration": i + 1,
                                 "wall_ms": ms, "prediction": prediction,
                                 "expected": case["expected"]})
        stats = {"median_ms": statistics.median(times), "p95_ms": percentile(times, .95),
                 "min_ms": min(times), "max_ms": max(times),
                 "correct_repeated_predictions": correct, "repeats": args.repeats}
        with RamSampler(process, interval) as ram:
            for _ in range(3):
                predict(text)
        case["warm_memory"] = ram.report()
        case["latency"] = stats
        print(f"Median {stats['median_ms']:.2f} ms | P95 {stats['p95_ms']:.2f} ms | "
              f"PASS {correct}/{args.repeats} | RSS peak {ram.peak / MIB:.1f} MiB", flush=True)
        report["cases"].append(case)
        write_csv(result_path(out, "latency.csv"), latency_rows)
        result_path(out, "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Complete all memory/latency measurements before any profiler allocation.
    for case in cases:
        text = case["text"]
        print(f"\nProfiling layer: {case['id']}", flush=True)
        for _ in range(args.warmup):
            predict(text)
        captured = []
        def capture(module, inputs, kwargs):
            captured.append({"args": tensor_shapes(inputs, torch),
                             "kwargs": tensor_shapes(kwargs, torch)})
        handle = agent.model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            inspected = predict(text)
        finally:
            handle.remove()
        case["model_forward_inputs"] = captured
        case["usage"] = inspected.get("usage")

        handles = install_module_scopes(agent.model, record_function)
        try:
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True,
                         profile_memory=True) as prof:
                for _ in range(args.profile_repeats):
                    with record_function("PREDICT::" + case["id"]):
                        predict(text)
        finally:
            for handle in handles:
                handle.remove()
        prof.export_chrome_trace(str(result_path(out, case["id"] + "_trace.json")))
        layer_rows = operation_rows(prof, case["id"], args.profile_repeats)
        all_layers.extend(layer_rows)
        aggregates = defaultdict(lambda: [0.0, 0, 0.0])
        for row in layer_rows:
            group = aggregates[(row["operation"], row["input_shapes"])]
            group[0] += row["self_cpu_ms"]
            group[1] += row["calls"]
            group[2] += row["share_of_aten_self_cpu_pct"]
        ops = [{"case": case["id"], "operation": op, "input_shapes": shape,
                "calls": value[1], "calls_per_inference": value[1] / args.profile_repeats,
                "self_cpu_ms": value[0],
                "self_cpu_ms_per_inference": value[0] / args.profile_repeats,
                "share_of_aten_self_cpu_pct": value[2]}
               for (op, shape), value in aggregates.items()]
        ops.sort(key=lambda x: x["self_cpu_ms"], reverse=True)
        all_ops.extend(ops)
        case["top_operations"] = ops[:10]
        case["top_layer_operations"] = layer_rows[:10]
        print("Operasi dominan (persen dari aten self CPU dalam profiler):")
        for row in ops[:5]:
            print(f"  {row['operation']:45} {row['share_of_aten_self_cpu_pct']:6.2f}% "
                  f"{row['input_shapes']}")
        print("Layer + operasi dominan:")
        for row in layer_rows[:3]:
            print(f"  {row['module']} | {row['operation']} | "
                  f"{row['share_of_aten_self_cpu_pct']:.2f}%")
        # Save each completed case so long runs retain useful progress.
        write_csv(result_path(out, "latency.csv"), latency_rows)
        write_csv(result_path(out, "operators.csv"), all_ops)
        write_csv(result_path(out, "layer_operations.csv"), all_layers)
        result_path(out, "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        # Release large profiler event lists before the next case's memory run.
        del prof
    print(f"\nSelesai. Hasil: {out.resolve()}")
    print(f"Parameter FP32: {report['model']['parameter_mib']:.1f} MiB")
    for phase, memory in report["memory"].items():
        print(f"{phase}: sampled RSS peak {memory['sampled_peak_rss_mib']:.1f} MiB; "
              f"OS lifetime peak {memory['process_lifetime_peak_rss_mib']:.1f} MiB")


if __name__ == "__main__":
    main()
