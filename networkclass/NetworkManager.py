import copy
import numpy as np
from . import RunningAverage,NestedAverage,NetworkManagerArea

class NetworkManager:
    def __init__(self,depth,branch_list=None,max_branch_id=None):
        self.last_branch_id = 0
        self.network = list()
        self.debug = list()
        self.label = list()
        self.label_coord = list()
        for i in range(depth):
            self.network.append(list())
            self.debug.append(list())
        self.branch_target = [0]
        if max_branch_id is not None:
            self.branch_list = copy.deepcopy(branch_list)
            self.branch_length = [0] * (max_branch_id+1)
            self.last_branch_id=max_branch_id
        else:
            self.branch_list = [0]
            self.branch_length = [0]
        self.branch_mean_radius = [RunningAverage()]
        self.branch_nested_intensity = [NestedAverage()]
        
    def get_new_branch(self,parent_branch_id=0,reverse=False):
        self.last_branch_id = self.last_branch_id+1
        self.branch_length.append(0)
        self.branch_target.append(0)
        self.branch_list.append(self.last_branch_id)
        self.branch_mean_radius.append(RunningAverage())
        if reverse:
            self.branch_nested_intensity.append(NestedAverage())
        else:
            self.branch_nested_intensity.append(self.branch_nested_intensity[parent_branch_id].copy())
        return self.last_branch_id
    
    def get_mean_radius(self,branch_id):
        return self.branch_mean_radius[branch_id]

    def get_nested_intensity(self,branch_id):
        return self.branch_nested_intensity[branch_id]

    def get_branch_target(self,branch_id):
        return self.branch_target[branch_id]

    def set_branch_target(self,branch_id,target):
        self.branch_target[branch_id] = (int(target[0]),int(target[1]))

    def append_to_network(self,depth,branch_id,contour,predicted=False):
        self.network[depth].append(NetworkManagerArea(branch_id,contour,predicted))
        self.branch_length[branch_id]+=1
    
    def append_to_debug(self,depth,contour):
        self.debug[depth].append(NetworkManagerArea(-1,contour,False))

    def add_label(self,coord,text):
        text_dict = {
    'string': text,
    'size': 10,
    'color': 'green',
    'translation': np.array([0, -10])
}
        self.label.append(text)
        self.label_coord.append(np.array(coord))

    def get_branchs(self):
        return self.branch_list

    def get_branch(self,branch_id):
        branch = list()
        offset = 0
        for i in reversed(range(len(self.network))):
            area = next((area for area in self.network[i] if area.branch_id == branch_id),None)
            if len(branch) == 0:
                offset = i
            if area != None:
                branch.append(area)
            else:
                if len(branch)>0:
                    break
        return branch,offset

    def get_last_from_branch(self,branch_id):
        for i in range(len(self.network)):
            area = next((area for area in self.network[i] if area.branch_id == branch_id),None)
            if area != None:
                return area

    def get_branch_length(self,branch_id):
        return self.branch_length[branch_id]

    def remove_branch(self,branch_id):
        found = False
        for i in reversed(range(len(self.network))):
            area = next((area for area in self.network[i] if area.branch_id == branch_id),None)
            if area != None:
                found = True
                self.network[i].remove(area)
            else:
                #La branche a été trouvée et on est arrivée au bout de sa supression
                if found:
                    break
        self.branch_length[branch_id] = 0
        self.branch_list.remove(branch_id)
    
    #Each subnetwork is a different image
    def generate3DImages(self,shape):
        generated_images = []
        print(self.branch_list)
        print(self.branch_length)
        for branch_id in self.branch_list:
            image = np.zeros(shape,dtype=np.uint8)
            branch,offset = self.get_branch(branch_id)
            for area in branch:
                for point in area.contour:
                    image[offset][point[0][1]][point[0][0]] = 255
                offset -= 1
            generated_images.append(image)
        #Display debug
        image = np.zeros(shape,dtype=np.uint8)
        for i in reversed(range(len(self.network))):
            for area in self.debug[i]:
                for point in area.contour:
                    image[i][point[0][1]][point[0][0]] = 255
        generated_images.append(image)
        return generated_images