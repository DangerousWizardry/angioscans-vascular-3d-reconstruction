from functools import partial

from matplotlib import pyplot as plt
from functions.contours_utils import contour_intersect, merge_contours
from functions.exploration_utils import *
from tqdm.notebook import tqdm
from . import *


class NetworkEngine:
    def __init__(self,image,max_depth):
        self.image = image
        self.max_depth = max_depth
        self.network_manager = NetworkManager(max_depth)
        self.stack = list()
        self.branch_depth = [max_depth-1]
        # if a branch has a value < current depth in this array, this branch is monitored until the registered depth to not merge with another one
        self.branch_monitored_depth = [max_depth]
        self.branch_protected_split = [max_depth]
        self.current_depth = max_depth-1
        self.alpha = 0.6 #alpha >= beta
        self.beta = 0.6 #similarity score
        self.stop_depth = 200
        self.pbar = tqdm(total=max_depth-self.stop_depth)

    def get_new_branch(self,parent_branch_id,reverse=False,force_depth=None):
        child_id = self.network_manager.get_new_branch(parent_branch_id,reverse)
        # Reversed exploring do not use branch_depth, branch_monitored_depth, branch_protected_split because it implies no splitting
        if reverse:
            self.branch_depth.append(0)
            self.branch_monitored_depth.append(0)
            self.branch_protected_split.append(0)
        else:
            if force_depth is None:
                self.branch_depth.append(self.branch_depth[parent_branch_id])
                self.branch_monitored_depth.append(self.max_depth)
                self.branch_protected_split.append(self.branch_depth[parent_branch_id])
            else:
                self.branch_depth.append(force_depth)
                self.branch_monitored_depth.append(force_depth)
                self.branch_protected_split.append(force_depth)
        return child_id
    
    def success_split(self,parent_branch,new_target,child_depth,exclusion_radius):
        self.network_manager.set_branch_target(parent_branch,new_target)
        print("SUCCESS SPLIT PARENT "+str(parent_branch))
        self.network_manager.branch_mean_radius[parent_branch] = RunningAverage(exclusion_radius,1,exclusion_radius)
        self.branch_monitored_depth[parent_branch] = child_depth
        self.branch_protected_split[parent_branch] = child_depth
        #Rollback branch 1 if success
        self.branch_depth[parent_branch]+=1
        self.network_manager.network[self.branch_depth[parent_branch]].remove(next((area for area in self.network_manager.network[self.branch_depth[parent_branch]] if area.branch_id == parent_branch),None))

    def explore(self,branch_id,iteration=1,excluded=None,onsuccess=None):
        multi = True if iteration > 1 else False
        for i in range(iteration):
            foundTarget = regionSearch(self.image[self.branch_depth[branch_id],:,:],self.network_manager.get_branch_target(branch_id),self.network_manager.get_mean_radius(branch_id).get_average(),self.network_manager.get_nested_intensity(branch_id).get_average()*self.alpha)
            if not isinstance(excluded,type(None)):
                #Case exploring a newly splitted branch
                tube_contour = regionGrowing(self.image[self.branch_depth[branch_id],:,:],foundTarget,self.network_manager.get_nested_intensity(branch_id),self.network_manager.get_mean_radius(branch_id).get_average(),self.beta,excluded)
            elif(self.branch_monitored_depth[branch_id]<self.branch_depth[branch_id]):
                #Case a splitted branch has succeed and we do not want it to merge with parent
                excluded = list()
                for area in self.network_manager.network[self.branch_depth[branch_id]]:
                    excluded.append(area.contour)
                tube_contour = regionGrowing(self.image[self.branch_depth[branch_id],:,:],foundTarget,self.network_manager.get_nested_intensity(branch_id),self.network_manager.get_mean_radius(branch_id).get_average(),self.beta,excluded)
            else:
                tube_contour = regionGrowing(self.image[self.branch_depth[branch_id],:,:],foundTarget,self.network_manager.get_nested_intensity(branch_id),self.network_manager.get_mean_radius(branch_id).get_average(),self.beta)
            if len(tube_contour)==0:
                self.branch_depth[branch_id] = 0
                if self.network_manager.branch_length[branch_id] < 20:
                    self.network_manager.remove_branch(branch_id)
                break
            center, radius = cv2.minEnclosingCircle(tube_contour)
            self.network_manager.add_label((self.branch_depth[branch_id],center[1],center[0]),
                                           str(round(radius/self.network_manager.get_mean_radius(branch_id).get_average(),2))+","+
                                           str(round(self.network_manager.get_mean_radius(branch_id).get_average(),2))+","+
                                           str(round(self.network_manager.get_nested_intensity(branch_id).get_average(),2)))
            
            if not multi and radius > 3 and self.branch_protected_split[branch_id] > self.branch_depth[branch_id] and (radius < self.network_manager.get_mean_radius(branch_id).get_average()*0.65 
                              or radius/self.network_manager.get_mean_radius(branch_id).get_last() < 0.75):
                print("Splitted "+str(branch_id)+" because protected until "+str(self.branch_protected_split[branch_id]))
                print("Found target is "+str(foundTarget)+" radius is "+str(radius))
                #We suspect a splitted vessel
                last_radius = self.network_manager.get_mean_radius(branch_id).get_last()
                #We try to fit an rectangle to get the split orientation
                tube_contour = self.network_manager.get_last_from_branch(branch_id).contour
                rect = cv2.minAreaRect(tube_contour)
                rect_points = cv2.boxPoints(rect).astype(int)
                # 2---------3 
                # |   RECT  | then offset_a
                # 1---------0
                # 2---------1 
                # |   RECT  | then offset_b
                # 3---------0
                offset_a = np.subtract(rect_points[1],rect_points[0]) #orientation vector
                offset_b = np.subtract(rect_points[3],rect_points[0])
                if np.sum(np.abs(offset_a)) > np.sum(np.abs(offset_b)):
                    offset = offset_a 
                else:
                    offset = offset_b
                pixel_1 = (rect[0] + offset/4).astype(int)
                pixel_2 = (rect[0] - offset/4).astype(int)
                rect_points = np.append(rect_points,pixel_1)
                rect_points = np.append(rect_points,pixel_2)

                self.network_manager.append_to_debug(self.branch_depth[branch_id],rect_points.reshape(6,1,2))

                exclusion_radius = max(rect[1])/4 #rect width
                #x_offset = int(last_radius/2)
                #pixel_1 = (self.network_manager.get_branch_target(branch_id)[0]-x_offset,self.network_manager.get_branch_target(branch_id)[1])
                #Build exclusion zone
                template = np.zeros((self.image.shape[1],self.image.shape[2]),np.uint8)
                cv2.circle(template,pixel_1,int(exclusion_radius),255)
                contours, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                exclusion_zone = contours #cv2.pointPolygonTest()
                
                #Not a split, branch just disappear
                if(exclusion_radius < 1):
                    self.branch_depth[branch_id] = 0
                    print(str(branch_id)+"stop because exclusion contours was "+str(contours))
                    if self.network_manager.branch_length[branch_id] < 20:
                        self.network_manager.remove_branch(branch_id)
                else:
                    self.network_manager.append_to_debug(self.branch_depth[branch_id],contours[0])
                    #pixel_2 = (self.network_manager.get_branch_target(branch_id)[0]+x_offset,self.network_manager.get_branch_target(branch_id)[1])
                    #Creating a new branch with a new target
                    target_2_branch_id = self.get_new_branch(branch_id)
                    print("Create branch id :"+str(target_2_branch_id))
                    self.network_manager.set_branch_target(target_2_branch_id,pixel_2)
                    self.network_manager.get_mean_radius(target_2_branch_id).add_number(exclusion_radius)
                    #Exploring 20 frames of new branch with a success callback for parent branch
                    self.stack.append([target_2_branch_id,20,exclusion_zone,partial(self.success_split, branch_id, pixel_1,self.branch_depth[branch_id]-20,exclusion_radius)])
                    #In case branch failed we set up parent branch to continue, these data will be rollback if child branch succeed
                    self.network_manager.append_to_network(self.branch_depth[branch_id],branch_id,tube_contour)
                    self.branch_depth[branch_id]-=1
                    self.network_manager.set_branch_target(branch_id,center)
            
            else:
                if radius == 0:
                    self.branch_depth[branch_id] = 0
                    if self.network_manager.branch_length[branch_id] < 20:
                        self.network_manager.remove_branch(branch_id)
                    break
                self.network_manager.append_to_network(self.branch_depth[branch_id],branch_id,tube_contour)
                self.branch_depth[branch_id]-=1
                self.network_manager.set_branch_target(branch_id,center)
                self.network_manager.get_nested_intensity(branch_id).add_number(self.image[self.branch_depth[branch_id],self.network_manager.get_branch_target(branch_id)[1],self.network_manager.get_branch_target(branch_id)[0]])
                self.network_manager.get_mean_radius(branch_id).add_number(radius)
        if multi and i == iteration-1:
            onsuccess()
        return None
    
    def explore_reverse(self,initial_point,initial_depth):
        branch_id = self.get_new_branch(-1,True)
        print(initial_point)
        print(self.image[initial_depth,initial_point[1],initial_point[0]])
        self.network_manager.set_branch_target(branch_id,initial_point)
        self.network_manager.get_mean_radius(branch_id).add_number(10)
        self.network_manager.get_nested_intensity(branch_id).add_number(self.image[initial_depth,initial_point[1],initial_point[0]])
        for depth in range(initial_depth,self.max_depth):
            print("Print target is "+str(self.network_manager.get_branch_target(branch_id))+ "for depth "+str(depth))
            foundTarget = regionSearch(self.image[depth,:,:],self.network_manager.get_branch_target(branch_id),self.network_manager.get_mean_radius(branch_id).get_average(),self.network_manager.get_nested_intensity(branch_id).get_average()*self.alpha)
            print("found target "+str(foundTarget))
            print("debug : avg intensity " + str(self.network_manager.get_nested_intensity(branch_id).get_average()) + " radius "+str(self.network_manager.get_mean_radius(branch_id).get_average()))
            print("target intensity :"+str(self.image[depth,foundTarget[1],foundTarget[0]]))
            tube_contour = regionGrowing(self.image[depth,:,:],foundTarget,self.network_manager.get_nested_intensity(branch_id),self.network_manager.get_mean_radius(branch_id).get_average(),self.beta)
            if len(tube_contour) == 0:
                if self.network_manager.branch_length[branch_id] < 20:
                    self.network_manager.remove_branch(branch_id)
                    return 0
                return self.network_manager.branch_length[branch_id]
            center, radius = cv2.minEnclosingCircle(tube_contour)
            if radius == 0:
                if self.network_manager.branch_length[branch_id] < 20:
                    self.network_manager.remove_branch(branch_id)
                    return 0
                return self.network_manager.branch_length[branch_id]
            self.network_manager.append_to_network(depth,branch_id,tube_contour)
            self.network_manager.set_branch_target(branch_id,center)
            self.network_manager.get_nested_intensity(branch_id).add_number(self.image[depth,self.network_manager.get_branch_target(branch_id)[1],self.network_manager.get_branch_target(branch_id)[0]])
            self.network_manager.get_mean_radius(branch_id).add_number(radius)
        return self.network_manager.branch_length[branch_id]

    def force_prepare_explore(self, initial_point, initial_depth, initial_radius=3):
        branch_id = self.get_new_branch(0,False,initial_depth)
        self.network_manager.set_branch_target(branch_id,initial_point)
        self.network_manager.get_nested_intensity(branch_id).add_number(self.image[initial_depth,initial_point[1],initial_point[0]])
        self.network_manager.get_mean_radius(branch_id).add_number(initial_radius)
        self.current_depth = initial_depth
        print("Force create branch "+str(branch_id)+" at depth "+str(initial_depth))

    def explore_all(self):
        for i in range(len(self.branch_depth)): 
            #print(str(self.network_manager.branch_mean_radius[i]))   
            if self.branch_depth[i] >= self.current_depth:
                for j in range(self.branch_depth[i] - self.current_depth + 1):
                    self.stack.extend([[i]])
        #print()
    def run(self):
        self.explore_all()
        while self.current_depth > self.stop_depth:#1120:
            while len(self.stack) > 0:
                exploring_parameters = self.stack.pop()
                self.explore(*exploring_parameters)
                #print(str(self.stack)+" : "+str(time.time()))
            self.current_depth -= 1
            self.pbar.update(1)
            self.pbar.set_description("Searching region %s" % str(self.branch_depth))
            self.explore_all()
        return

    def merge_network(self):
        net = self.network_manager.network
        merged_branch_list = list()
        merged_to_branch_list = list()
        result_network_manager = NetworkManager(len(self.image),self.network_manager.branch_list,self.network_manager.last_branch_id)

        for depth in range(len(net)):
            nodes = net[depth]
            to_merge = dict()
            for ida in range(len(nodes)):
                to_merge[ida] = set()
            for ida in range(len(nodes)):
                contour = list()
                if nodes[ida].branch_id in merged_branch_list:
                    merged_to_id = merged_to_branch_list[merged_branch_list.index(nodes[ida].branch_id)]
                    merge_to_local_idxx = next((i for i, item in enumerate(nodes) if item.branch_id == merged_to_id), None)
                    if merge_to_local_idxx is not None:
                        to_merge[next((i for i, item in enumerate(nodes) if item.branch_id == merged_to_id), None)].add(ida)
                    else:
                        # Branch has been merge to a branch that is not there anymore
                        # Then we cancel the merge from this depth to the top 
                        merged_branch_list_idx = merged_branch_list.index(nodes[ida].branch_id)
                        merged_to_branch_list.pop(merged_branch_list_idx)
                        merged_branch_list.pop(merged_branch_list_idx)
                else:
                    for idb in range(len(nodes)):
                        if ida < idb :
                            if contour_intersect(nodes[ida].contour,nodes[idb].contour):
                                merged_branch_list.append(nodes[idb].branch_id)
                                merged_to_branch_list.append(nodes[ida].branch_id)
                                to_merge[ida].add(idb)
                                branch_to_force_merge = list()
                                for i in range(len(merged_to_branch_list)):
                                    if(nodes[idb].branch_id == merged_to_branch_list[i]):
                                        merged_to_branch_list[i] = nodes[ida].branch_id
                                        branch_to_force_merge.append(merged_branch_list[i])
                                
                                for branch in range(len(branch_to_force_merge)):
                                    local_branch_idb = next((i for i, item in enumerate(nodes) if item.branch_id == branch), None)
                                    if local_branch_idb is not None:
                                        to_merge[ida].add(local_branch_idb)
                                
                                print(str(nodes[idb].branch_id)+" merged to "+str(nodes[ida].branch_id))
                
                for key, value in to_merge.items():
                    if len(value) == 0:
                        if nodes[key].branch_id not in merged_branch_list:
                            result_network_manager.append_to_network(depth,nodes[key].branch_id,nodes[key].contour)
                    else:
                        contours_to_merge = [node.contour for index,node in enumerate(nodes) if index in value]
                        contours_to_merge.append(nodes[key].contour)
                        #debug_id = [node.branch_id for index,node in enumerate(nodes) if index in value]
                        #print("at depth"+str(depth)+" merge "+str(nodes[key].branch_id)+" with"+str(debug_id))
                        contour = merge_contours(self.image[0],contours_to_merge)
                        result_network_manager.append_to_network(depth,nodes[key].branch_id,contour)
        for branch_id in result_network_manager.branch_list:
            if result_network_manager.branch_length[branch_id] < 30:
                result_network_manager.remove_branch(branch_id)
        return result_network_manager
    
    def segmentize(self,network):
        #for i in range()
        return False