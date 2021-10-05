import pickle
import os

"""
We split 'train_data.pickle' to fit the upload file size.
reference : https://towardsdatascience.com/how-to-spit-a-pickled-model-file-to-bypass-upload-limits-on-pythonanywhere-e051ea1cec2d
"""

def split(source, dest_folder, write_size):

    partnum = 0
    input_file = open(source, 'rb')
    while True:
        chunk = input_file.read(write_size)

        if not chunk:
            break

        partnum += 1
        filename = 'BPO/train_data_bpo.pkl' + str(partnum)
        dest_file = open(filename, 'wb')
        dest_file.write(chunk)
        dest_file.close()
    input_file.close()
    return partnum


def join(source_dir, dest_file, read_size):
    output_file = open(dest_file, 'wb')

    parts = ['BPO/train_data_bpo.pkl1', 'BPO/train_data_bpo.pkl2', 'BPO/train_data_bpo.pkl3']

    for file in parts:

        path = file

        input_file = open(path, 'rb')

        while True:
            bytes = input_file.read(read_size)
            if not bytes:
                break
            output_file.write(bytes)

        input_file.close()

    output_file.close()


join(source_dir='', dest_file="BPO/train_data_bpo.pkl", read_size=100000000)

#split(source='BPO/train_data_bpo.pkl', write_size=100000000, dest_folder='BPO')
