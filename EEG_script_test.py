import numpy as np
import sys
import glob
import os
import export_epoch_to_nifti_small
import numpy as np
import platform
import random

try:
    import mne
except ImportError as error:
    sys.path.append("/home/nas/PycharmProjects/mne-python/")
    import mne

import mne_bids

node = platform.node()
if "beluga" in node:
    data_path = "/home/knasioti/projects/rrg-gdumas85/data/HBN/EPO/"
    export_folder = '/home/knasioti/projects/rrg-gdumas85/knasioti/test_BIDS'
    fake_raw_fname = "/home/knasioti/projects/rrg-gdumas85/data/HBN/RAW.fif"


elif "acheron" in node:
    data_path = "/home/nas/Consulting/ivadomed-EEG/HBN/EPO/"
    export_folder = '/home/nas/Desktop/test_BIDS'
    fake_raw_fname = "/home/nas/Consulting/ivadomed-EEG/RAW.fif"

else:
    raise NameError("need to specify a path where the HBN dataset is stored")



subjects = [os.path.basename(x) for x in glob.glob(os.path.join(data_path, "*"))]

# Utilize info from the raw files - THIS IS NOT IMPLEMENTED FROM GUILLAUME'S FILES - FAKE IT
fake_raw = mne.io.read_raw_fif(fake_raw_fname)

# Assign line frequency - Required for BIDS export
#fake_raw.info['line_freq'] = 60

annotated_event_for_gt = '998'  # This is the event that will be used to create the derivatives
                                    # 999 Heartbeats
                                    # 998 Blinks

# Randomize subjects - this will help when submitting multiple jobs in Beluga

# Select channel type to create the topographies on
ch_type = 'eeg'

n_skipped_subjects = 0
iSubject = 1
for subject in subjects:

    frontalEOGChannel = 'E22'
    # Load preprocessed mne data from Guillaume
    fname = os.path.join(data_path, subject, 'RestingState_Blinks_epo.fif')
    epochs = mne.read_epochs(fname, proj=True, preload=True, verbose=None)

    if len(epochs) < 50:  # Only take into account the cases when the algorithm "PROBABLY" detects blinks correctly
        subject = subject.lower() + "01"  # BIDS compliance

        #eo.plot_image(picks=[frontalEOGChannel])

        ## Do some preprocessing on the epochs - jitter the center so the model doesn't learn the position
        epochs_preprocessed = epochs.crop(tmin=-0.4+np.random.random()*0.1, tmax=0.40+np.random.random()*0.1, include_tmax=True)

        # Check if resampling is needed
        epochs_preprocessed = epochs_preprocessed.resample(100)

        # Create bids folder
        bids_path = mne_bids.BIDSPath(subject=subject, root=export_folder)

        # Use the raw object that the trials came from in order to build the BIDS tree
        mne_bids.write_raw_bids(fake_raw, bids_path, overwrite=True, verbose=True)

        # Export trials into .nii files
        export_epoch_to_nifti_small.run_export(epochs_preprocessed, ch_type, annotated_event_for_gt, bids_path)
        print("Just finished subject: " + str(iSubject))
        print("Already skipped: " + str(n_skipped_subjects) + " subjects")
        iSubject += 1
    else:
        n_skipped_subjects += 1
        print("Just finished subject: " + str(iSubject))
        print("Already skipped: " + str(n_skipped_subjects) + " subjects")
