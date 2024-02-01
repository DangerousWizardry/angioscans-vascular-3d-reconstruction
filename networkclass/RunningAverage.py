import copy

# This class is used to perform a fast average on a growing-only list
class RunningAverage:
    def __init__(self,sum=0,count=0,last=None):
        self.sum = sum
        self.count = count
        self.last = last
        

    def add_number(self, number):
        self.sum += number
        self.last = number
        self.count += 1

    def get_average(self):
        if self.count == 0:
            return 0 
        return self.sum / self.count
    
    def split(self):
        if self.count > 0:
            self.sum /= 4
            self.count /= 2
        return self
    
    def get_last(self):
        return self.last
    
    def copy(self):
        return copy.deepcopy(self)
    
    def __str__(self) -> str:
        return str(self.get_average())+" ("+str(self.count)+")"