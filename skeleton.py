import csv
import glob
import os
import sys
from copy import deepcopy
from typing import Optional
import random
import pandas as pd
import numpy as np

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    # print(result[0]['age']) # debug: testing.
    return result

def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True



class DGHNode(object):
    
    def __init__(self, name='name', children=None, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        if children is not None:
            for c in children:
                self.insertChild(c)
        
    def insertChild(self, child):
        self.children.append(child)
        child.parent = self
        
    def isRoot(self):
        return (self.parent is None)
    
    def depth(self):   
        if self.isRoot():
            return 0
        else:
            return 1 + self.parent.depth()
        
    def isLeaf(self):
        return (len(self.children) == 0)
    
    def descendantCount(self):
        dCount = 0
        for c in self.children:
            if c.isLeaf():
                dCount += 1
            else:
                dCount = dCount + c.descendantCount()
        return dCount


class DGHTree(object):
    
    def __init__(self):
        self.root = None
        self.nodes = []
        
    def insertNode(self, child, parent):
        if parent is None:
            if self.root is None:
                self.root = child
        else:
            parent.insertChild(child)
        self.nodes.append(child)
    
    def search(self, nodeName):  
        index = -1
        for node in self.nodes:
            index += 1
            if node.name == nodeName:
                return index
            
    def leafCount(self):
        leaves = 0
        for node in self.nodes:
            if node.isLeaf():
                leaves += 1
        return leaves


def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    #TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.
    tree = DGHTree()
    with open(DGH_file, "r") as f:
        lines = f.readlines()
        parentNodeIndex = 0
        prevTabCount = 0
        for l in lines:
            line = ' '.join(l.split())
            if l != '\n' or not l.strip():
                tabCount = 0
                for char in l.rstrip():
                    if char == '\t':
                        tabCount += 1
                newNode = DGHNode(line)
                if tabCount == 0:
                    #root
                    tree.insertNode(newNode, None)
                else:
                    if prevTabCount < tabCount:
                        tree.insertNode(newNode, tree.nodes[-1])
                    elif prevTabCount == tabCount:
                        tree.insertNode(newNode, tree.nodes[-1].parent)
                    else:
                        loopNo = prevTabCount - tabCount
                        p = tree.nodes[-1]
                        for i in range(loopNo+1):
                            parent = p.parent
                            p = parent
                        tree.insertNode(newNode, p)
            prevTabCount = tabCount
    return tree


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file);
    return DGHs



def removeSensitiveAttribute(dataset, DGHs):
    datasetWithoutSA = deepcopy(dataset)
    QI_set = set(DGHs.keys())
    attribute_set = set(dataset[0].keys())
    set_difference = attribute_set.difference(QI_set)
    SA = list(set_difference)[0]
    for record in datasetWithoutSA:
        record.pop(SA, None)
    return datasetWithoutSA


def random_divider(dataset, k):
    ordered_list = list(enumerate(dataset))
    random.shuffle(ordered_list)
    indices, records = zip(*ordered_list)
    index_list = list(indices)
    divided = [index_list[i * k:(i + 1) * k] for i in range((len(index_list) + k - 1) // k )] 
    counter = 0
    for i in divided:
        if len(i) < k:
            divided[counter-1].extend(i)
            divided.remove(i)
        counter += 1
    return divided


#put each record attribute into set
#if length is bigger than 1, needs to generalize
#check attribute matching
#if attributes are not the same, generalize

def generalization(db, k, DGHs):
    dataset = deepcopy(db)
    divided_indices = random_divider(dataset, k)
    generalizedDB = generalize_ec(dataset, divided_indices, DGHs)
    return generalizedDB
                
def generalize_ec_into_setlist(dataset, ec, DGHs):
    datasetWithoutSA = removeSensitiveAttribute(dataset, DGHs)
    ec_list = [set() for _ in range(len(datasetWithoutSA[0]))]
    for r in ec:
        record = datasetWithoutSA[r]
        for i in range(len(record)):
            ec_list[i].add(list(record.values())[i])
    return ec_list

def generalize_up(dataset, ec_count, attribute_count, divided, DGHs):
    for i in divided[ec_count]:
        record = dataset[i]
        attribute_name = list(record)[attribute_count]
        attribute_value = record[attribute_name]
        #print("{}: {}".format(attribute_name, attribute_value))
        dgh_attribute_tree = DGHs[attribute_name]
        dghNodeIndex = dgh_attribute_tree.search(attribute_value)
        dghNode = dgh_attribute_tree.nodes[dghNodeIndex]
        if not dghNode.isRoot():
            dghParent = dghNode.parent
            update_dict = {attribute_name : dghParent.name}
            #print(update_dict)
            dataset[i].update(update_dict)
    return dataset
    
def generalize_ec(dataset, divided, DGHs):
    generalizedDB = None
    for ec_count, ec in enumerate(divided):
        #for each equivalence class
        ec_list = generalize_ec_into_setlist(dataset, ec, DGHs)
        for attribute_count, ec_set in enumerate(ec_list):
            #for each attribute
            if len(ec_set) != 1:
                #needs generalization
                #print(ec_set)
                generalizedDB = generalize_up(dataset, ec_count, attribute_count, divided, DGHs)
                #check again with the new DB
    if check_generalization_need(generalizedDB, ec, DGHs):
        generalize_ec(generalizedDB, divided, DGHs)
    return generalizedDB

def check_generalization_need(dataset, ec, DGHs):
    checkList = []
    ec_list = generalize_ec_into_setlist(dataset, ec, DGHs)
    for attribute_count, ec_set in enumerate(ec_list):
        if len(ec_set) != 1:
            checkList.append(ec_set)
    if not checkList:
        #checklist is empty, no more generalization needed
        return False
    else:
        #more generalization needed
        return True


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    
    anonymized_dataset = generalization(raw_dataset, k, DGHs)
    #TODO: complete this function.
    write_dataset(anonymized_dataset, output_file)


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    dataset_without_SA = removeSensitiveAttribute(raw_dataset, DGHs)
    cost = 0
    for data_count, raw_data in enumerate(dataset_without_SA):
        anon_data = anonymized_dataset[data_count]
        for attribute_count in range(len(raw_data)):
            attribute_raw = list(raw_data.values())[attribute_count]
            attribute_anon = list(anon_data.values())[attribute_count]
            if attribute_raw != attribute_anon:
                #print("generalization cost")
                attribute_name = list(raw_data)[attribute_count]
                attribute_dghTree = DGHs[attribute_name]
                raw_attr_index = attribute_dghTree.search(attribute_raw)
                anon_attr_index = attribute_dghTree.search(attribute_anon)
                raw_attr_node = attribute_dghTree.nodes[raw_attr_index]
                anon_attr_node = attribute_dghTree.nodes[anon_attr_index]
                raw_attr_depth = raw_attr_node.depth()
                anon_attr_depth = anon_attr_node.depth()
                depth_difference = raw_attr_depth - anon_attr_depth
                cost += depth_difference
    return float(cost)


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)
    
    #TODO: complete this function.
    datasetWithoutSA = removeSensitiveAttribute(anonymized_dataset, DGHs)
    attributeCount = len(datasetWithoutSA[0])
    recordCount = len(datasetWithoutSA)
    attributeMatrix = np.zeros((attributeCount, recordCount)).T
    #print(attributeMatrix)
    for i, record in enumerate(datasetWithoutSA):
        for j, attributeDict in enumerate(record):
            attributeName = list(record)[j]
            attributeValue = list(record.values())[j]
            dgh = DGHs[attributeName]
            nodeIndex = dgh.search(attributeValue)
            node = dgh.nodes[nodeIndex]
            attributeMatrix[i][j] = calculateLMCost(node, dgh)
    #sum over all attributes
    LM_cost_rec = attributeMatrix.sum(axis=1)
    #sum over all records
    LM_cost_table = sum(LM_cost_rec) / attributeCount
    return LM_cost_table

def calculateLMCost(node, dgh):
    valueCost = 0
    if node.isRoot():
        valueCost += 1
    elif node.isLeaf():
        valueCost += 0
    else:
        numerator = node.descendantCount()
        denominator = dgh.leafCount()
        valueCost += (numerator - 1) / (denominator - 1)
    return valueCost


def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    
    #TODO: complete this function.
    datasetWithoutSA = removeSensitiveAttribute(raw_dataset[0:100], DGHs)
    #initially records in datasetWithoutSA are unmarked
    ec_divided = []
    for record in datasetWithoutSA:
        record['marked'] = 0
        record['equivalence-class'] = 0
   
    equivalence_class = 0
    
    for record_count, record in enumerate(datasetWithoutSA):
        if record['marked'] == 0:
            if checkUnmarkedRecordCount(datasetWithoutSA, k):
                equivalence_class += 1
                rec = datasetWithoutSA[record_count]
                rec['marked'] = 1
                rec['equivalence-class'] = equivalence_class
                #Find the k-1 unmarked records from D that are closest to metric dist
                ec_record_index_list = calculateDistanceList(datasetWithoutSA, record_count, k, DGHs)
                ec_record_index_list.append(record_count)
                ec_divided.append(ec_record_index_list)
                for ind in ec_record_index_list:
                    toBeMarked = datasetWithoutSA[ind] 
                    toBeMarked['marked'] = 1
                    toBeMarked['equivalence-class'] = equivalence_class
            else:
                break
                   
    #if j>0 unmarked records remain in D then
    #merge these records with the records in the Ec that was constructed last
    #create a (k+j) anonymous EC from these records
    #end if
    for record_count, record in enumerate(datasetWithoutSA):
         if record['marked'] == 0:
                ec_divided[-1].append(record_count)
                record['equivalence-class'] = equivalence_class
                record['marked'] = 1
                
    print(ec_divided)
    
    anonymized_dataset = generalize_ec(raw_dataset, ec_divided, DGHs)
    
    write_dataset(anonymized_dataset, output_file)
    
def checkUnmarkedRecordCount(db, k):
    unmarkedCount = 0
    for record in db:
        if record['marked'] == 0:
            unmarkedCount += 1
    #print(unmarkedCount)
    return unmarkedCount >= k

def calculateDistanceList(datasetWithoutSA, record_count, k, DGHs):
    ec_index_list = []
    rec = datasetWithoutSA[record_count]
    for c, record in enumerate(datasetWithoutSA):
        #print("-------------Starting Calculation for {}------------------".format(c))
        if c != record_count:
            if record['marked'] == 0:
                tup = (c, calculateDist(rec, record, DGHs))
                
                ec_index_list.append(tup)
                #print(ec_index_list)
    ec_index_list.sort(key=lambda tup: tup[1])
    #print(ec_index_list)
    indx_list = [a_tuple[0] for a_tuple in ec_index_list[0:k-1]]
    
    return indx_list

def calculateDist(record1, record2, DGHs):
    rec1 = record1.copy()
    rec2 = record2.copy()
    new_recs = []
    #print([rec1, rec2])
    keys_to_remove = ['marked', 'equivalence-class']
    for key in keys_to_remove:
        rec1.pop(key, None)
        rec2.pop(key, None)
    attr_list = createAttributeSetList(rec1, rec2)
    generalization_cost = 0
    for attribute_count, ec_set in enumerate(attr_list):
            #print("----------NEW ATTRIBUTE---------------------------")
            #print("{}: {}".format(attribute_count, ec_set))
            #for each attribute
            if len(ec_set) != 1:
                #needs generalization
                gener_cost, new_recs = generalizeRecords(rec1, rec2, attribute_count, DGHs)
                rec1, rec2 = new_recs[0], new_recs[1]
                #print("------FINISH----------")
                #print(new_recs)
                #print("----------------")
                generalization_cost +=  gener_cost
                #print(generalization_cost)
                
                #if gener_check(new_recs, attribute_count):
                    #generalization_cost = generalization_cost + calculateDist(new_recs[0], new_recs[1], DGHs)
    if gener_check(rec1, rec2):
        generalization_cost = generalization_cost + calculateDist(rec1, rec2, DGHs)
    return generalization_cost

def createAttributeSetList(rec1, rec2):
    attr_list = [set() for _ in range(len(rec1))]
    for i in range(len(rec1)):
        attr_list[i].add(list(rec1.values())[i])
        attr_list[i].add(list(rec2.values())[i])
    #print(attr_list)
    return attr_list

def generalizeRecords(rec1, rec2, attribute_count, DGHs):
    cost = 0
    record_list = [rec1.copy(), rec2.copy()]
    gen_list = []
    #print("------START----------")
    #print(record_list)
    for record in record_list:
        attribute_name = list(record)[attribute_count]
        attribute_value = record[attribute_name]
        #print("{}: {}".format(attribute_name, attribute_value))
        dgh_attribute_tree = DGHs[attribute_name]
        dghNodeIndex = dgh_attribute_tree.search(attribute_value)
        dghNode = dgh_attribute_tree.nodes[dghNodeIndex]
        if not dghNode.isRoot():
            dghParent = dghNode.parent
            update_dict = {attribute_name : dghParent.name}
            record.update(update_dict)
            cost += 1
        gen_list.append(record)
   
    return cost, gen_list

def gener_check(rec1, rec2):
    checkList = []
    attr_list = createAttributeSetList(rec1, rec2)
    
    for attribute_count, ec_set in enumerate(attr_list):
        if len(ec_set) != 1:
            checkList.append(ec_set)
    if not checkList:
        #checklist is empty, no more generalization needed
        return False
    else:
        #more generalization needed
        return True


def topdown_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Top-down anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    
    #TODO: complete this function.
    dataset_without_SA = removeSensitiveAttribute(raw_dataset, DGHs)
    specialization_nodes = {}
    #create the root of the specialization tree
    original_index_list = list(range(len(dataset_without_SA)))
    for dgh in DGHs:
        dghTree = DGHs[dgh]
        specialization_nodes[dgh] = dghTree.root
        
    #start top_down from root
    ec_list = find_s_star(dataset_without_SA, DGHs, [specialization_nodes], [original_index_list], k)
    anonymized_dataset = generalize_ec(raw_dataset, ec_list, DGHs)
    write_dataset(anonymized_dataset, output_file)
    
            
def find_s_star(db, DGHs, s, o, k):
    for i in range(len(s)):
        ec_divided = []
        specialization_nodes = s[i]
        original_index_list = o[i]
        
        #root
        attribute_lm_cost, tuple_list = calculate_next_s_star(db, DGHs, specialization_nodes, original_index_list)
        
        ec_list, new_specialization_nodes_list = divide_into_ec(attribute_lm_cost, specialization_nodes, tuple_list, k)
          
        ec_divided.extend(ec_list)
            
    #find_s_star(db, DGHs, new_specialization_nodes_list, ec_list, k)
    return ec_divided

def divide_into_ec(attribute_lm_cost, specialization_nodes, tuple_list, k):
    new_specialization_nodes_list = []
    s = min(attribute_lm_cost, key=attribute_lm_cost.get)
    print(s)
    ec_list = []
    children = []
    if attribute_lm_cost:
        #find the children of s
        for sp in specialization_nodes[s].children:
            children.append(sp.name)
            new_dict = specialization_nodes.copy()
            new_dict[s] = sp
            new_specialization_nodes_list.append(new_dict)

        for count, c in enumerate(children):
            n = [a_tuple[1] for a_tuple in tuple_list if a_tuple[2]==c]
            if n:
                if does_not_violate_k(n, k):
                    ec_list.append(n)
                else:
                    attribute_lm_cost.pop(s, None)
                    divide_into_ec(attribute_lm_cost, specialization_nodes, tuple_list, k)  
    else:
        print('stop branching for this tuple list')
        return ec_list, []
 
    return ec_list, new_specialization_nodes_list
    

def calculate_next_s_star(db, DGHs, specialization_nodes, original_index_list):
    s = ''
    tuple_list = []
    attribute_lm_cost = {}
    for dgh in DGHs:
        if can_split_node(specialization_nodes[dgh]):
            dghTree = DGHs[dgh]
            lm_value = 0
                   
            for data_index in original_index_list:
                record = db[data_index]
                attr_node_index = dghTree.search(record[dgh])
                attr_node = dghTree.nodes[attr_node_index]
                #print("Attribute value: {}".format(attr_node.name))

                while generalization_check(attr_node, specialization_nodes[dgh]):
                    #print('needs more generalizing up')
                     attr_node = generalize_one_node_up(attr_node)

                        #print("{}".format(attr_node.name))
                tuple_list.append((dgh ,data_index, attr_node.name))

                        #calculate lm 
                lm_value += calculateLMCost(attr_node, dghTree)

                attribute_lm_cost[dgh] = lm_value
    #print(tuple_list)
    return attribute_lm_cost, tuple_list
                
def can_split_node(node):
    return not node.isLeaf()

def does_not_violate_k(lst, k):
    return len(lst) >= k 

def generalization_check(attr_node, spc):
    if attr_node.parent.name == spc.name:
            return False
    return True

def generalize_one_node_up(node):
    return node.parent



# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k")
    print(f"\tWhere algorithm is one of [clustering, random, topdown]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'topdown']:
    print("Invalid algorithm.")
    sys.exit(2)

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer");
function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")


# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300

