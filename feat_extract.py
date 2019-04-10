import os
import pandas as pd
from collections import Counter
import sys
from matplotlib import pyplot as plt

three_to_one = {
        "ala" : "A",
        "arg" : "R",
        "asn" : "N",
        "asp" : "D",
        "cys" : "C",
        "glu" : "E",
        "gln" : "Q",
        "gly" : "G",
        "his" : "H",
        "ile" : "I",
        "leu" : "L",
        "lys" : "K",
        "met" : "M",
        "phe" : "F",
        "pro" : "P",
        "ser" : "S",
        "thr" : "T",
        "trp" : "W",
        "tyr" : "Y",
        "val" : "V"}

def get_feats(main_path="",
              infile_name="data/train.csv",
              infile_all="data/all.mfa",
              expasy_path="expasy/",
              assign_class=0,
              exract_locs=[(0,5),(-5,None),(0,10),(-10,None),(0,20),(-20,None),
                           (0,1),(1,2),(2,3),(3,4),(4,5),(-1,None),(-2,-1),(-3,-2),(-4,-3),(-5,-4)],
              order_aa=[]):

    libs_prop = get_libs_aa(os.path.join(main_path,expasy_path))
    feats = list(libs_prop.keys())

    if len(order_aa) == 0:
        seqs_all = get_all_seqs(os.path.join(main_path,infile_all))
        order_aa = get_all_poss_aa(seqs_all)
        del seqs_all

    seqs = get_all_seqs(os.path.join(main_path,infile_name))

    data_dict_simple = get_feats_simple(seqs,libs_prop,assign_class=assign_class)     

    seq_feats_fragments = {}
    for ident,seq in seqs.items():
        instance_dict = {}
        seq_splits = [seq[ex_loc_start:ex_loc_stop] for ex_loc_start,ex_loc_stop in exract_locs]
        seq_splits

        instance_dict["seq_length"] = len(seq)

        instance_dict.update(dict(zip(order_aa,count_aa(seq,aa_order=order_aa))))
        
        for index,spl_s in enumerate(seq_splits):
            temp_dict = get_feats_simple_seq(spl_s,libs_prop)
            temp_dict = dict([[k+"|"+str(index),v] for k,v in temp_dict.items()])
            instance_dict.update(temp_dict)
        
        seq_feats_fragments[ident] = instance_dict

    # TODO more checks before combining
    # now that we have everything extracted lets check integrety
    assert len(seq_feats_fragments) == len(data_dict_simple.keys()), \
           "Length of fragments exrtracted features for (%s) not equal to simple feature extraction step (%s)" % (len(seq_feats_fragments),len(data_dict_simple))
    assert len(seqs.keys()) == len(data_dict_simple.keys()), \
           "Length of sequences extracted (%s) not equal to simple feature extraction step (%s)" % (len(seqs),len(data_dict_simple))
    # TODO add not only the seq feat libs, but also seq length and count vector
    #assert len(exract_locs) == int(len(seq_feats_fragments[0].keys())/len(libs_prop.keys())), \
    #       "Length of features extracted for sequence locations (%s) not equal to sequence splits (%s)" % (len(exract_locs),int(len(seq_feats_fragments[0].keys())/len(libs_prop.keys())))
    
    ret_df = {}
    for ident,seq in seqs.items():
        # TODO repair same identifier problem
        # This WILL overwrite sequences with the same identifiers
        ret_df[ident] = {}
        try:
            ret_df[ident].update(data_dict_simple[ident])
        except KeyError:
            # big problem sys exit here. feat extract went wrong
            sys.exit(42)
        try:
            ret_df[ident].update(seq_feats_fragments[ident])
        except KeyError:
            # big problem sys exit here. feat extract went wrong
            sys.exit(42)
        
    ret_df = pd.DataFrame(ret_df).T
    
    return ret_df

def count_substring(string,sub_string):
    count=0
    for i in range(len(string)-len(sub_string)+1):
        if(string[i:i+len(sub_string)] == sub_string ):      
            count += 1
    return count 
        
def split_seq(a, n):
    k, m = divmod(len(a), n)
    return(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def get_all_poss_aa(seqs):
    poss_aa = set()
    for ident,seq in seqs.items():
        for aa in Counter(seq).keys():
            poss_aa.add(aa)
    return list(poss_aa)

def count_aa(seq,aa_order=[]):
    feat_vector = []
    counted_aa = Counter(seq)
    for aa in aa_order:
        try: feat_vector.append(counted_aa[aa])
        except: feat_vector.append(0)
    return feat_vector

def get_count_aa(seq,aa_to_count):
    total = 0
    for aa in aa_to_count:
        total += seq.count(aa)
    return total

def apply_prop_lib(seq,lib,avg=True):
    ret_val = 0.0
    for aa in seq:
        try: ret_val += lib[aa]
        except KeyError:
            # TODO write to stderr
            print("------ apply_prob_lib() ------\n")
            print("Error could not find corresponding library value for: %s" % (aa))
            print("For dictionary: \n %s \n" %  (lib))
            print("------------------------------\n")
            continue
    if avg: 
        try: ret_val = ret_val/float(len(seq))
        except ZeroDivisionError: 
            ret_val = 0.0
            # TODO write to stderr
            print("-----------")
            print("\nError sequence has length of zero\n")
            print("-----------")
    return ret_val

def analyze_lib(infile_name):
    infile = open(infile_name)
    prop_dict = {}
    for line in infile:
        line = line.strip()
        if len(line) == 0: continue
        aa_three,val = line.lower().split(": ")
        val = float(val)
        prop_dict[three_to_one[aa_three]] = val
    return prop_dict

def get_set_all_aa(infile_name,ret_duplicated_pos=False,skip_line_startswith="identifier"):
    infile = open(infile_name)
    if ret_duplicated_pos: locs = []
    else: locs = set()
    gene_ident_to_row_ident = {}

    for line in infile:
        if line.startswith(skip_line_startswith): continue

        splitline = line.split(",")
        ident = splitline[1]
        start_pos = int(splitline[7])
        end_pos = int(splitline[8])
        
        if ident in gene_ident_to_row_ident.keys():
            gene_ident_to_row_ident[ident].append(splitline[0])
        else:
            gene_ident_to_row_ident[ident] = [splitline[0]]
        
        for pos in range(start_pos,end_pos+1):
            if ret_duplicated_pos: locs.append("%s|%s" % (ident,str(pos)))
            else: locs.add("%s|%s" % (ident,str(pos)))
    
    return locs,gene_ident_to_row_ident

def get_set_all_aa_compare(infile_name,locs_other,return_agg_seq=True,min_overlap_agg=0.95,skip_line_startswith="identifier"):
    infile = open(infile_name)
    num_in_other = 0
    num_total = 0
    distrib_overlap = []
    agg_seqs = []
    
    for line in infile:
        if line.startswith(skip_line_startswith): continue
        
        splitline = line.split(",")
        ident = splitline[1]
        seq = splitline[10]
        start = int(splitline[7])
        end = int(splitline[8])
        
        in_depleted = False
        tot_overlap = 0
        agg_seq = ""
        indexes_aa = []

        for index,pos in enumerate(range(start,end)):
            index_aa = int((index - (index % 3))/3)
            ident_loc = "%s|%s" % (ident,str(pos))
            if ident_loc in locs_other:
                in_depleted = True
                tot_overlap += 1
            else:
                if index_aa in indexes_aa: continue
                indexes_aa.append(index_aa)
                agg_seq += seq[index_aa]
        if tot_overlap/float(len(range(start,end+1))) > min_overlap_agg:
            agg_seqs.append(list(zip(agg_seq,indexes_aa)))
        distrib_overlap.append(tot_overlap/float(len(range(start,end+1))))
        num_total += 1
        if in_depleted:    num_in_other += 1
    return distrib_overlap,agg_seqs,num_total,num_in_other
    
def get_libs_aa(dirname):
    path = dirname
    listing = os.listdir(path)
    libs_prop = {}
    for infile in listing:
        if not infile.endswith(".txt"): continue
        libs_prop["".join(infile.split(".")[:-1])] = analyze_lib(os.path.join(path,infile))
    return libs_prop

def get_all_seqs(infile_name,skip_line_startswith="#"):
    try:
        infile = open(infile_name)
    except IOError:
        # TODO write to stderr
        print("Error: could not find the following file to get sequences at the following path: %s" % (infile_name))
        sys.exit(41)
                
    seqs = {}
    ident = "unidentified:class"
    seq = ""

    for line in infile:
        if line.startswith(skip_line_startswith): continue
        if line.startswith(">"):
            if len(seq) > 0:
                seqs[ident] = seq
            ident = line.rstrip().lstrip(">")
            seq = ""
        else:
            seq += line.rstrip()

    return seqs

def get_feats_simple_seq(seq,libs_prop):
    new_instance = {}
    for name,lib in libs_prop.items():
        new_instance[name] = apply_prop_lib(seq,lib)
    return(new_instance)

def extract_rolling_features(seq,lib,lib_name="feat",num_aas=[2,4,8,16,20,25],percentiles=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]):
    ret_dict = {}

    feature_seqs = pd.Series([lib[aa] for aa in seq if aa in lib.keys()])
    
    for naa in num_aas:
        seq_feat = feature_seqs.rolling(naa).sum()
        perc_feat = dict([("percentile_%s_%s_%s" % (naa,p,lib_name),seq_feat.quantile(p)) for p in percentiles])
        ret_dict.update(perc_feat)
        
    return(ret_dict)

def get_feats_simple(seqs,libs_prop,assign_class=1,nmer_feature="data/nmer_features.txt",add_rolling_feat=True):
    nmers = map(str.strip,open(nmer_feature).readlines())

    data_df = {}

    for ident,seq in seqs.items():
        new_instance = {}
        for name,lib in libs_prop.items():
            new_instance[name] = apply_prop_lib(seq,lib)
            
            if add_rolling_feat:
                new_instance.update(extract_rolling_features(seq,lib,lib_name=name))
        new_instance["class"] = assign_class
        
        for nmer in nmers:
            new_instance["countnmer|"+nmer] = count_substring(seq,nmer)

        data_df[ident] = new_instance
    return data_df