# Thermal Face Contrastive GAN

Needs a reorg of the entire repo.

##TFC-GAN-FFT

All models for the ICIP paper, where I only show FFT-GLO-16 and FFT-PATCH-16

For dissertation, I need to add all the results even for Patch-4.

For NeurIPS using TFC-GAN-STN, I can show the results (when unaligned) on the Devcom_5Perc dataset also showing SSIM scores.

##STN

Use `TFCGAN_STN21_Original_NewModel3_Official.py` as the best performing STN. This is where there are two `fake_A1` and `fake_A2`.

Use `TFCGAN_STN21_Eur_DarkVisible.py` when faces are unlit as in the case of Eurecom. In this case, there is only one `fake_A`.
