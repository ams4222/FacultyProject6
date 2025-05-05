#!/usr/bin/env python3
import os
import sys
import pickle
from scapy.all import *
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP
from multiprocessing import Pool
import json

"""
When this is run, it will recursively search through directories to find
PCAP files. For each PCAP file, parse through grabbing timestamps and 
checking if packet is outgoing or incoming.
After PCAP is fully parsed, output to a text file (or otherwise specified)
Continue with the next PCAP until no other PCAP.
Should combine all the txt files into one directory
"""

DO_TSHARK_FILTERING = False
IP_TARGET_DEFAULT = []
BATCH_IPS = []
COMBINED_SEQUENCES = []  # Global list to store all sequences


def parse_pcap_ip(path, adjust_times=True, ip_targets=IP_TARGET_DEFAULT):
    """
    function processes IP-level capture pcap into a sequence of 2-tuple packet representations.
    the Scapy library is used for parsing the captures
    """
    if ip_targets is None:
        ip_targets =  IP_TARGET_DEFAULT
    sequence = []
    packets = rdpcap(path)
    start_time = None
    ips = set()
    for packet in packets:
        if IP in packet:
            ips.add(packet[IP].dst)
            ips.add(packet[IP].src)

            direction = None
            if packet[IP].dst in ip_targets:
                direction = -1
            elif packet[IP].src in ip_targets:
                direction = 1

            if not direction:
                continue

            timestamp = packet.time
            # save initial start time
            if start_time is None:
                start_time = timestamp
            length = len(packet)

            # add to sequence
            sequence.append((timestamp, direction * length))

    # adjust packet times such that the first packet is at time 0
    if adjust_times and start_time:
        sequence = [(pkt[0] - start_time, pkt[1]) for pkt in sequence]

    return sequence


def save_to_file(sequence, path):
    """Save sequence to an individual pickle file and add to global combined list"""
    global COMBINED_SEQUENCES
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    # Save individual file
    with open(path + '.pkl', 'wb') as file:
        pickle.dump(sequence, file)

    # Append to global combined list (with filename for reference)
    COMBINED_SEQUENCES.append((os.path.basename(path), sequence))


def parse_task(filepath):
    """function to handle the processing of a single pcap file, for IP-level captures"""
    try:
        root, fname = filepath
        path = os.path.join(root, fname)
        batch, _, _, inst = fname.replace('.pcap','').split("_")
        batch, inst = int(batch), int(inst)
        ip_targets = BATCH_IPS[batch]
        if DO_TSHARK_FILTERING:
            # use tshark to filter out noise packets (reduces computation time for some datasets)
            new_path = filter_pcap(path, ip_targets)
        else:
            new_path = path

        # use Scapy to process pcap into packet sequence
        sequence = parse_pcap_ip(new_path, ip_targets=ip_targets)

        # remove filtered captured
        if new_path.endswith(".ttmp"):
            os.remove(new_path)

        return sequence, os.path.join(*filepath)
    except Exception as exc:
        print("encountered exception", exc, os.path.join(*filepath))
    return None, None


def filter_pcap(filepath, ip_targets):
    """
    use tshark to filter traces by IP/MAC address
    :return: the path to the filtered pcap file
    """
    # generate pathname for temporary filtered pcap
    new_path = os.path.join(os.path.dirname(filepath), os.path.basename(filepath), ".ttmp")

    # wireshark filter to select only relevant packets
    tfilter = ' or '.join(["ip.addr == {}".format(addr) for addr in ip_targets])

    # run tshark filtering
    os.system("tshark -r {file} -q -2 -R \"{filter}\" -w {outpath} {tail}".format(file=filepath,
                                                                                  filter=tfilter,
                                                                                  outpath=new_path,
                                                                                  tail="2>/dev/null"))
    return new_path


def preprocessor(inpath, output):
    """
    Start a multiprocessing pool to handle processing pcap files in parallel.
    Packet sequences are saved to a text file following Wang's format as the worker processes produce results.
    The site names are mapped to numbers dynamically, and these mappings are saved for later reference.
    This function will load prior mappings if a file is provided.
    :param input: root directory path containing pcap files
    :param output: directory which to save trace files
    :param site_map: path to file where site to number mappings should be saved
    :return: nothing
    """
    # create list of pcap files to process
    flist = []
    for root, dirs, files in os.walk(inpath):
        # filter for only pcap files
        files = [fi for fi in files if fi.endswith(".pcap")]
        flist.extend([(root, f) for f in files])

    try:
        # process pcaps in parallel
        with Pool() as pool:
            procs = pool.imap_unordered(parse_task, flist)

            # iterate through processed pcaps as they become available
            # pcaps are parsed in parallel, however parsed sequences are saved to file in serial
            for i, res in enumerate(procs):
                print("Progress: {}/{}                \r".format(i + 1, len(flist)), end="")

                # save the sequence to file
                sequence, filepath = res[0], res[1]
                if sequence is not None:
                    # save to file
                    out_path = os.path.join(output, os.path.basename(filepath).replace('.pcap', ''))
                    save_to_file(sequence, out_path)

    except KeyboardInterrupt:
        print("Caught keyboard interrupt. Doing cleanup...")

    # lazy make directories
    try:
        os.makedirs(os.path.dirname(site_map))
        os.makedirs(os.path.dirname(instance_map))
        os.makedirs(os.path.dirname(checkpoint))
    except:
        pass
    # Save combined pickle file
    combined_path = os.path.join(output, 'all_sequences.pkl')
    with open(combined_path, 'wb') as file:
        pickle.dump(COMBINED_SEQUENCES, file)
    print(f"\nCombined sequences saved to {combined_path}")


def parse_arguments():
    import argparse
    """parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--INPUT",
                        required=True)
    parser.add_argument("--OUTPUT",
                        required=True)
    parser.add_argument("--IP_TARGETS", 
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    BATCH_IPS = []
    with open(args.IP_TARGETS, 'r') as fi:
        for line in fi:
            line = line.strip()
            BATCH_IPS.append(list(line.split(' ')))
    #seq, fpath = parse_task(('deepcorr_crawler/inflow', '0_1_0_0.pcap'))
    preprocessor(args.INPUT,
                 args.OUTPUT)
