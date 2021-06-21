import numpy as np
import sys
import glob
import os
import export_epoch_to_nifti_small
import numpy as np
import platform
import random
import pandas as pd
import csv
import json

try:
    import mne
except ImportError as error:
    sys.path.append("/home/nas/PycharmProjects/mne-python/")
    import mne

import mne_bids

node = platform.node()
if "beluga" in node:
    data_path = ""
    export_folder = '/home/knasioti/projects/rrg-gdumas85/knasioti/test_BIDS'

elif "acheron" in node:
    data_path = "/home/nas/Consulting/ivadomed-EEG/MEG data/"
    export_folder = '/home/nas/Desktop/test_BIDS_singleSubject'

    if not os.path.exists(export_folder):
        os.mkdir(export_folder)
else:
    raise NameError("need to specify a path where the HBN dataset is stored")

meg4_files = glob.glob(data_path + "/**/*.meg4", recursive=True)
meg4_files = [x for x in meg4_files if "hz.meg4" not in x]


# Create a dataframe with all the information regarding the studies
df_dataset = pd.DataFrame()
studies = []
subjects = []
sessions = []
runs = []
foldernames = [os.path.dirname(x) for x in meg4_files]
for file in meg4_files:
    if "sub-" not in file:  # Regular CTF file structure
        studies.append(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file)))))
        subjects.append(os.path.basename(os.path.dirname(os.path.dirname(file))))
        sessions.append(os.path.basename(os.path.dirname(file)))
        runs.append(file.split("_")[-3] + file.split("_")[-1].split(".meg4")[0])
    else:  # BIDS compliant
        studies.append(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file)))))))
        subjects.append(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file))))))
        sessions.append(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file)))))
        runs.append(file.split("_")[-2])
df_files = pd.DataFrame({"Study": studies, "Subject": subjects, "Session": sessions, "Run": runs, "Foldername": foldernames, "Filename": meg4_files}, columns=["Study", "Subject", "Session", "Run", "Foldername", "Filename"])


# Select channel type to create the topographies on
ch_type = 'mag'  # grad or mag - The CTF data have only mag???

time_around_event_for_trial = 0.5  # +-

# Assign a new subject on each session
df_files['Subject'] = ['a' + str(i) for i in range(len(df_files))]


# Loop across recordings
for index, df_recording in df_files.iterrows():

    subject = df_recording['Subject'].lower().replace('sub-', '')
    session = df_recording['Run'].lower().replace('run-', '')

    try:
        bids_path = mne_bids.BIDSPath(subject=subject, session=session, root=export_folder)
    except:
        print('a')

    if not os.path.exists(str(bids_path.directory)):

        # Get events
        #events = mne.read_events('/home/nas/Consulting/ivadomed-EEG/MEG data/Reverse Correlation/20150529/subject3_visual_20150529_01.ds/subject3_visual_20150529_01.eve')

        #if df_recording['Filename'] == '/home/nas/Consulting/ivadomed-EEG/MEG data/Reverse Correlation/20140718/pilot3_visual_20140718_01.ds/pilot3_visual_20140718_01.meg4':

        #if "20140718" in df_recording['Filename']:
        events_file = df_recording['Filename'].replace(".meg4", ".eve")  # The assumption here is that the blinks have been exported as the sole event in the .eve format

        if os.path.exists(events_file):
            events = mne.read_events(events_file)
            raw = mne.io.read_raw_ctf(df_recording['Foldername'], preload=True)


            # Reject events that are very close to each other. This is done so there are no more than 1 blink within the same trial
            a = np.diff(events[:, 0])
            indices_to_keep = np.insert(np.logical_or(a < 2, a > time_around_event_for_trial*raw.info["sfreq"]), 0, True)
            events = events[indices_to_keep, ]

            #a = mne.compute_proj_raw(raw, start=0, stop=None, duration=1, n_grad=2, n_mag=2, n_eeg=0, reject=None,
            #                     flat=None, n_jobs=1, meg='separate', verbose=None)
            #raw.add_proj(a)

            # Read epochs
            #epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.5, tmax=0.5, baseline=(-0.5, 0.5), proj=True, preload=True)  # proj=True,
            epochs = mne.Epochs(raw, events, event_id=1,
                                tmin=-time_around_event_for_trial, tmax=time_around_event_for_trial,
                                baseline=(None, None), proj=True, preload=True, picks=["data"])
            annotated_event_for_gt = '1'  # This is the event that will be used to create the derivatives

            #mne.compute_proj_epochs(epochs, n_grad=2, n_mag=2, n_eeg=2, n_jobs=1, desc_prefix=None, meg='separate',
            #                        verbose=None)[source]

            #a= mne.compute_proj_epochs(epochs, n_grad=2, n_mag=2, n_eeg=2, n_jobs=1, desc_prefix=None, meg='separate',
            #                        verbose=None)


            # Make a raster plot from selected channels for evaluation
            #epochs.plot_image(picks=['MLT32-4409', 'VEOG'])


            # ALWAYS RESAMPLE - THE 3RD DIMENSION ON THE MODEL NEEDS A STANDARDIZED SAMPLERATE
            epochs_preprocessed = epochs.resample(100)

            # Export trials into .nii files
            export_epoch_to_nifti_small.run_export(epochs_preprocessed, ch_type, annotated_event_for_gt, bids_path, df_recording)


        else:
            print('No event file created for file: ' + df_recording['Filename'])

    else:
        print("Already converted " + df_recording['Filename'])

    # Add tsv entry
    participants_file = os.path.join(str(bids_path.root), 'participants.tsv')

    if os.path.exists(participants_file):
        previous_participants = pd.read_csv(participants_file, sep=',')
        subs = previous_participants['participant_id'].tolist()
        subs.append('sub-' + subject)
        sexes = previous_participants['sex'].tolist()
        sexes.append('na')
        ages = previous_participants['age'].tolist()
        ages.append('na')

        df_participants = pd.DataFrame({'participant_id': subs,
                                        'sex': sexes,
                                        'age': ages},
                                        columns=['participant_id', 'sex', 'age'])

    else:
        df_participants = pd.DataFrame({'participant_id': ['sub-'+subject], 'sex': ['na'], 'age': ['na']},
                                       columns=['participant_id', 'sex', 'age'])

    df_participants.to_csv(participants_file, sep=',', index=False)


# Write the rest files
with open(str(bids_path.root) + '/README', 'w') as readme_file:
    readme_file.write('MEG Dataset for ivadomed.')

data_json = {"participant_id": {
    "Description": "Unique ID",
    "LongName": "Participant ID"
},
    "sex": {
        "Description": "M or F",
        "LongName": "Participant sex"
    },
    "age": {
        "Description": "yy",
        "LongName": "Participant age"
    }
}
with open(str(bids_path.root) + '/participants.json', 'w') as json_file:
    json.dump(data_json, json_file, indent=4)

# Dataset description in the output folder
dataset_description = {"BIDSVersion": "1.6.0",
                       "Name": "ms_challenge_2021"
                       }


with open(str(bids_path.root) + '/dataset_description.json', 'w') as json_file:
    json.dump(dataset_description, json_file, indent=4)

# Dataset description in the derivatives folder
dataset_description = {"Name": "Example dataset",
                       "BIDSVersion": "1.6.0",
                       "PipelineDescription": {"Name": "Example pipeline"}
                       }

with open(str(bids_path.root) + '/derivatives/dataset_description.json', 'w') as json_file:
    json.dump(dataset_description, json_file, indent=4)
