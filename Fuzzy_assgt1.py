import numpy as np
import skfuzzy as fuzz
import re
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm

#read problem file and parse

class Rules:
    def __init__(self,fuzzy_set,rb):
        self.ruleset = list()
        for i in rb:
            if i != 'name':
                self.ruleset.append(Rule(rb[i],i))
        #print(self)
    def __repr__(self):
        str = list()
        for i in self.ruleset:
            str.append(i.__repr__())
        str = '\n'.join(str)
        return str

class Rule:
    def __init__(self,rule,name):
        self.rulename = name
        self.values = list()
        self.connectives = list()
        self.out = list()
        line  = rule.lower().strip('if')
        line = line.strip('.').split()

        #print(line)
        for i in range(len(line)):
            if line[i].lower() == 'and' or line[i].lower() == 'or':
                #print(line[i])
                self.connectives.append(line[i])
                #print(i)

        line  = rule.lower().strip('if')
        line = line.strip('.').strip()
        line = re.split('then',line)

        self.out = line[1]
        split = list()
        temp = dict()
        split = re.split('is',self.out)
        a = split[0].strip()
        b = split[1].strip()
        temp[a] = b
        self.out = temp

        line = re.split('and | or ',line[0])
        for i in line:
            split = list()
            temp = dict()
            split = re.split('is',i)
            a = split[0].strip()
            b = split[1].strip()
            temp[a] = b
            self.values.append(temp)
        #print(line)
            #print(i)
            #if line[i] == 'If' or line[i] == 'if':
        #print(self)
    def __repr__(self):
        return (self.rulename + ': Values: '+str(self.values)+ ' Connectives: '+ str(self.connectives)+ ' Outputs: '+str(self.out))

class Fuzzy_problem_solver:
    def __init__(self,file):
        self.pars = Parser(file)
        self.mfx = dict()
        self.mfx_out = dict()
        self.rules = Rules(self.pars.fuzzy_sets,self.pars.rule_base)
        #print(self.rules)
        self.set_up_mfx()
        #self.calc_out()

        #print(self.mfx)
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def find_nearest_idx(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    def trap_vec(self,ranges,arr):
        #print(ranges)
        #print(arr)
        out = np.zeros(len(ranges))
        a = arr[0]-arr[2]
        b = arr[0]
        c = arr[1]
        d = arr[1]+arr[3]
        #print(a,b,c,d)
        if b-a != 0.:
            slope_1 = 1/(b-a)
            slope_1_c = 1. - slope_1*b
        if c-d != 0:
            slope_2 = 1/(c-d)
            slope_2_c = 0. - slope_2*d

#        print(slope_1,slope_2,slope_1_c,slope_2_c)
        slope_1_indexes = [self.find_nearest(ranges,a),self.find_nearest(ranges,b)]
        top_index = [self.find_nearest(ranges,b),self.find_nearest(ranges,c)]
        slope_2_indexes = [self.find_nearest(ranges,c),self.find_nearest(ranges,d)]

        #print(slope_1_indexes,top_index,slope_2_indexes)
        if b-a != 0.:
            for i in range(slope_1_indexes[0],slope_1_indexes[1]):
                out[i] = ranges[i]*slope_1 + slope_1_c
        elif (b-a) == 0:
            for i in range(slope_1_indexes[0],slope_1_indexes[1]):
                out[i] = 1.
        if c-d != 0:
            for i in range(slope_2_indexes[0],slope_2_indexes[1]+1):

                out[i] = ranges[i]*slope_2 + slope_2_c
        elif (c-d) == 0:
            for i in range(slope_2_indexes[0],slope_2_indexes[1]+1):
                out[i] = 1

        for i in range(top_index[0],top_index[1]):
            out[i] = 1.0

        #print(ranges,out)
        # plt.figure(figsize=(8, 5))
        # plt.plot(ranges,out)
        # plt.show()
        return out

    def interp_memb(self, range , y_vals, measurement):
        # arr_copy = range.copy()
        # a = self.find_nearest_idx(arr_copy,measurement)
        # #print(b)
        # if a <= measurement:
        #     b = a + 1
        # else:
        #     b = a
        #     a = a - 1
        # out = y_vals[a] + (measurement - range[a])*((y_vals[b]-y_vals[a])/(range[b]-range[a]))
        out = np.interp(measurement,range,y_vals)
        #print(measurement,out)
        return out

    def set_up_mfx(self):
        for i in self.pars.fuzzy_sets:
            #print(i,self.pars.fuzzy_sets[i])
            temp_dict = dict()
            for j in self.pars.fuzzy_sets[i]:
                mini = 0
                maxi = -10000000

                if maxi < self.pars.fuzzy_sets[i][j][1]:
                    maxi = self.pars.fuzzy_sets[i][j][1]
                if mini > self.pars.fuzzy_sets[i][j][0]:
                    mini = self.pars.fuzzy_sets[i][j][0]
            temp_range = np.arange(mini,maxi+1, 1)
            for j in self.pars.fuzzy_sets[i]:
                arr = self.pars.fuzzy_sets[i][j]
                # a = arr[0]-arr[2]
                # b = arr[0]
                # c = arr[1]
                # d = arr[1]+arr[3]
                # temp_dict[j] = [fuzz.trapmf(temp_range,(a,b,c,d)),temp_range]
                temp_dict[j] = [self.trap_vec(temp_range, arr),temp_range]
            self.mfx[i] = temp_dict

        for set in self.mfx:
            if set not in list(self.pars.measurments.keys()):
                #rint(set, list(self.measurments.keys()))
                self.mfx_out[set] = self.mfx[set]
        #print(self.mfx_out)
        #print(self.mfx)
    def calc_out(self,method = 'centroid', inputs = 'def'):

        rule_outputs = []
        no_act = True
        for rule in self.rules.ruleset:
            #print(rule.rulename)
            ruleout = []
            for value in rule.values:
                key = list(value)[0]
                key_range = value[key]
                if inputs == 'def':
                    #interp_membership
                    #print(self.mfx[key][key_range][1], self.mfx[key][key_range][0], self.pars.measurments[key])
                    tempo = self.interp_memb(self.mfx[key][key_range][1], self.mfx[key][key_range][0], self.pars.measurments[key])
                    #print(self.interp_memb(self.mfx[key][key_range][1], self.mfx[key][key_range][0], self.pars.measurments[key]))
                else:
                    #print(self.mfx[key][key_range][1], self.mfx[key][key_range][0], inputs[key])
                    #print(self.interp_memb(self.mfx[key][key_range][1], self.mfx[key][key_range][0], self.pars.measurments[key]))
                    tempo = self.interp_memb(self.mfx[key][key_range][1], self.mfx[key][key_range][0], inputs[key])


                ruleout.append(tempo)
                #print(self.pars.measurments[key],key,key_range,tempo)
            if len(rule.connectives) != 0:
                for conn in rule.connectives:
                    if conn.lower() == 'and':
                        rule_outputs.append([min(ruleout),rule.out])
                    else:
                        rule_outputs.append([max(ruleout),rule.out])
                #print(ruleout)
            elif len(rule.connectives) == 0:
                #print(ruleout)
                rule_outputs.append([ruleout,rule.out])
                #print(rule_outputs)

        #for each rule output
        out_dict = dict()

        for o,out_val in rule_outputs:
            #for each output type
            for out in self.mfx_out:
                out_dict[out]= dict()
                #for each fuzzy classification
                for out_val in self.mfx_out[out]:
                    out_dict[out][out_val]= []
        #print(rule_outputs)

        for real_out in rule_outputs:
            for z in real_out[1]:
                out_dict[z][real_out[1][z]].append(real_out[0])
        #print(out_dict)
        for real_out in out_dict:
            #print(real_out)
            for z in out_dict[real_out]:
                #print(out_dict[real_out][z])
                if max(out_dict[real_out][z]) != 0:
                    no_act = False
                out_dict[real_out][z] = max(out_dict[real_out][z])

        agg = ()
        leng = ()
        for output in out_dict:
            #agg[output] = list()
            for key in out_dict[output]:
                #print(out_dict[output][key],key)

                temp = np.clip(self.mfx_out[output][key][0], a_min = 0, a_max = (out_dict[output][key]))

                if len(agg) == 0:
                    leng = self.mfx_out[output][key][1]
                    agg = temp
                else:
                    agg = np.fmax(agg,temp)
        if no_act:
            return 0,out_dict
        #output_value = fuzz.defuzz(leng,agg,method)
        if method == 'centroid':
            output_value = np.dot(leng, agg) / np.sum(agg)
        else:
            output_value = 0
        #print(centroid, centroid- output_value)
        #5print(agg)
        #print(output_value)

        # plt.figure(figsize=(8, 5))
        # plt.plot(leng, self.mfx_out['d']['very small'][0], 'k')
        # plt.plot(leng, self.mfx_out['d']['small'][0], 'r')
        # plt.plot(leng, self.mfx_out['tip']['small'][0], 'b')
        # plt.plot(leng, self.mfx_out['tip']['moderate'][0], 'r')
        # plt.plot(leng, self.mfx_out['tip']['big'][0], 'g')
        # plt.plot(leng, agg, 'y')
        # plt.show()

        return output_value,out_dict
class Parser:
    def __init__(self,filename):
        with open(filename) as file:
            self.file_contents = file.read()
            self.parse()
    def parse(self):
        #print(type(self.file_contents))
        self.rule_base = dict()
        self.fuzzy_sets = dict()
        self.measurments = dict()
        self.outputs = dict()
        curr_set = 0
        empty_lin = 0
        Ruless = False
        for line in self.file_contents.splitlines():

            if line != '':
                if '=' not in line and 'Rule ' not in line and empty_lin == 0 and not Ruless:
                    self.rule_base['name']= line
                elif 'Rule ' in line and not Ruless:
                    temp_line = line.split()
                    separator = ' '
                    self.rule_base[(temp_line[0]+' '+temp_line[1].strip(':'))] = separator.join(temp_line[2:])

                elif '=' in line:
                    line = line.strip()
                    temp_line = line.split('=')
                    self.measurments[temp_line[0].strip().lower()] = float(temp_line[1])

                elif len(line.split()) == 1 and '=' not in line:

                    self.fuzzy_sets[line.strip().lower()] = dict()
                    curr_set = line.strip().lower()
                else:
                    temp = line.split()
                    #print(temp)
                    temp_list = temp[1:]
                    for i in range(len(temp_list)):
                        temp_list[i] = int(temp_list[i])
                    self.fuzzy_sets[curr_set][temp[0].lower()] = temp_list

            else:
                if empty_lin > 1  and not Ruless:
                    Ruless == True
                empty_lin +=1
        #print(self.fuzzy_sets)



# parser = Parser('example.txt')
fps = Fuzzy_problem_solver('example.txt')
#print(fps.calc_out())
print(fps.calc_out(inputs = {'journey_time':12, 'driving':45}))
X = np.arange(0, 20, 20/100)
Y = np.arange(0, 100, 100/100)
# X = np.arange(0, 10, 10/50)
# Y = np.arange(0, 10, 10/50)
# X = np.arange(40, 120, 120/100)
# Y = np.arange(0, 12, 12/100)
Z = list()
for i in Y:
    for j in X:
        z,_ = fps.calc_out(inputs = {'journey_time': j, 'driving': i})

        #z,_ = fps.calc_out(inputs = {'hr': i, 'r': j})
        #print(i,j,z)
        Z.append(z)

X, Y = np.meshgrid(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Z = np.array(Z).reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
#
plt.show()






#print(parser.measurment/s)
