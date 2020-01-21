import itertools
import random
import csv
import pandas as pd
from pathlib import Path

def create_ciri():
    ciri1 = []
    ciri2 = []
    ciri3 = []
    dna = ["A","C","G","T"]
    for i in dna:
        for j in dna:
            ciri1 +=[i+j]
            for k in dna:
                ciri2+=[i+j+k]
                for l in dna:
                    ciri3+=[i+j+k+l]
    ciri = ciri2 + ciri3 + ciri1
    return ciri;

def create_ciri5mers():
    ciri = []
    dna = ["A","C","G","T"]
    # for i in (5):
    ciri = [''.join(p) for p in itertools.product(dna, repeat=5)]
    return ciri;

def read_fasta_sequence(filename):
    # Read 1 byte. eg, '>'
    first_char = filename.read(1)
    # If it's empty, we're done.
    if (first_char == ""):
        return (["", ""])
    # If it's ">", then this is the first sequence in the file.
    elif (first_char == ">"):
        line = ""
    else:
        line = first_char
    # print(line)
    # Read the rest of the header line.
    line = line + filename.readline()
    # Get the rest of the ID. split to array
    words = line.split()
    if (len(words) == 0):
        print("No words in header line (%s)\n" % line)
    else:
        # id =words[7].replace('"','')
        id = words[1].split("=")[1]

    # Read the sequence, through the next ">".
    first_char = filename.read(1)
    sequence = ""
    while ((first_char != ">") and (first_char != "")):
        if (first_char != "\n"):  # Handle blank lines.
            line = filename.readline()
            sequence = sequence + first_char + line
        first_char = filename.read(1)

    # Remove EOLs.
    clean_sequence = ""
    for letter in sequence:
        if (letter != "\n"):
            clean_sequence = clean_sequence + letter
    sequence = clean_sequence

    # Remove space
    clean_sequence = ""
    for letter in sequence:
        if (letter != ' '):
            clean_sequence = clean_sequence + letter
    sequence = clean_sequence.upper()
    return id, sequence

def count_Arinimers(read):
    for i in range(len(ciri)):
        match = 0
        for k in range(3, 6):
            num_kmers = len(read) - k + 1
            for j in range(num_kmers):
                kmer = read[j:j+k]
                if len(kmer) == 5:
                    if kmer[0] + kmer[4] == ciri[i]:
                        match += 1
                        #counts[ciri[i]] = match
                        # print('cocok',kmer)
                else:
                    if kmer == ciri[i]:
                        match = match + 1
                        #counts[ciri[i]] = match
        counts[ciri[i]] = match
    #normalize it
    # for i in counts:
    #     counts[i] = counts[i]/len(read)
    return counts

def count_5mers(read):
    for i in range (len(ciri)):
        match = 0
        for k in range(5, 6):
            num_kmers = len(read) - k + 1
            for j in range(num_kmers):
                kmer = read[j:j+k]
                if kmer == ciri[i]:
                    match = match + 1
                        #counts[ciri[i]] = match
        counts[ciri[i]] = match
    #normalize it
    # for i in counts:
    #     counts[i] = counts[i]/len(read)
    return counts

#buat ciri
ciri = create_ciri()
counts = {}.fromkeys(ciri,[]) #buat dictionary untuk isi ciri

print(ciri)

# baca data latih dan uji
myfile = open("Sinorhizobium.fna", "r+")
# Print the title row.


with open('hasil_336 Arinimers_1000Sinorhizobium.csv', 'w+') as f:
    w = csv.DictWriter(f, counts.keys())
    w.writeheader()
    # read first seq from file
    [id, sequence] = read_fasta_sequence(myfile)

    # iterate until we've read the whole file
    i_sequence = 1
    while (id != ""):
        # Tell the user what's happening.
        if (i_sequence % 100 == 0):
            print("Reading %dth sequenza .\n" % i_sequence)
        vector = count_Arinimers(sequence)
        # vector['id_seq'] = id
        w.writerow(vector)
        # Read the next sequence.
        [id, sequence] = read_fasta_sequence(myfile)
        # print(id)
        i_sequence += 1
        # Close the file.
        i = 0

myfile.close()

