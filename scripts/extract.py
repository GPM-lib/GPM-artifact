import sys
import os
import getopt
import random
#import pandas as pd

'''
    extract rules: select `num` edges, collect their associated vertices, for undirected unweighted graph, add symmetry edges.
    write output into *.ext file.
'''

gpath = os.getcwd()
num = 0

def main(argv):
    global gpath, num
    try:
        opts, args = getopt.getopt(sys.argv[1:], "g:n:h", ['graph=','num=','help'])
    except getopt.GetoptError as e:
        usage()
        print('Got an error and exit, error is %s.' % str(e))
        sys.exit()

    for opt, arg in opts:
        if opt in ['-h','--help']:
            usage()
            sys.exit()
        elif opt in ['-g','--graph']:
            gpath = arg
            #print(gpath)
        elif opt in ['-n','--num']:
            num = int(arg)
            print("Extract {} edges.".format(num))
        else:
            print("Now %s" %str(opt))
            print("Error: Invaild parameters.")
            usage()
            sys.exit()
    edge_extract(gpath)
    
def edge_extract(graph_path):
    '''
    If we only choose vertex, if the chosed vertices are not connected, the edge list may be very small.
    If we add the associated vertex into the random_v list, the vertex list may be expanded.
    num shouldn't be too small.
    '''
    global num
    vertex_num, edge_num = counting(graph_path)
    o_v_num = o_e_num = 0
    all_edge = []
    print("vertex:{} edge: {}".format(vertex_num, edge_num))

    # select num vertices to reserve
    # random_v = random.sample(range(0, vertex_num), num)
    # print(random_e)
    f = open(graph_path,'r',encoding='utf-8')
    o = open(graph_path+".ext",'w',encoding='utf-8')
    lines = f.readlines()
    

    for line in lines:
        vedata = line.strip().split(" ")
        if vedata[0] == 't':
            o.write(line)
        elif vedata[0] == 'v':
            # check if vertex in random_v
            # do nothing
            pass
        elif vedata[0] == 'e':
            # check if edge contains any vertex in random_e
            # add the all edges to list
            # if len(vedata) == 4:
            all_edge.append(line)
        else:
            print('Error: Invalid graph file input.')
            sys.exit()

    # random select some edges
    # need to add symmetry edges for undirected graph
    extract_e = random.sample(all_edge, num)
    extract_e = add_symmetry(extract_e)

    # generate the associate vertices
    ass_ver = set()
    for i in extract_e:
        # try:
            e = i.strip().split(' ')
            ass_ver.add(e[1])
            ass_ver.add(e[2])
        # except:
        #     print(i)
    
    # construct vertices list  
    ass_ver = list(ass_ver)
    ass_ver = list(map(int, ass_ver))
    ass_ver.sort()
    print(ass_ver)
    ### o.writelines(ass_ver)
    for v in ass_ver:
        o.write("v " + str(v) + " 0\n")
    # construct edges list
    o.writelines(extract_e)

    # record the output info
    o_v_num = len(ass_ver)
    o_e_num = len(extract_e)
    print("Extract by edge, output vertex:{} edge: {}".format(o_v_num, o_e_num))
    f.close()
    o.close()

# outdated
def takeSecond(elem):
    try:
        return elem[0]
    except :
        print(len(elem))

def add_symmetry(edge_line):
    checked_list = []
    for info in edge_line:
        e = info.strip().split(" ")
        checked_list.append((e[1], e[2]))
    for info in edge_line:
        e = info.strip().split(" ")
        k = (e[2],e[1])
        if not k in checked_list:
            edge_line.append("e " + k[0] + " "+ k[1] + " 0\n")
    return edge_line

def counting(graph_path):
    vertex_count = 0
    edge_count = 0
    print("working in %s" % graph_path)
    f = open(graph_path)
    print("open file success.")

    while 1:
        lines = f.readlines(10000) # handle large file
        if not lines:
            break
        for line in lines:
            type = line.strip().split(" ")
            if type[0] == 'v':
                vertex_count += 1
            elif type[0] == 'e':
                edge_count += 1
            else:
                pass
    f.close()
    return vertex_count, edge_count

    
def usage():
    print("""
    Random extract some vertices with their edges.
    Function                        |   Usage
    Help                            |   -h
    input a graph                   |   -g graph_file_path 
    input edges number to reserve   |   -n number_of_edges
    """)


if __name__ == '__main__':
	main(sys.argv)